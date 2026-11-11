# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy
import functools
import logging
import math
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn

from megatron.core._rank_utils import log_single_rank
from megatron.core.extensions.transformer_engine import te_general_gemm
from megatron.core.fp8_utils import get_fp8_disabled_context
from megatron.core.models.common.embeddings import (
    RotaryEmbedding,
    YarnRotaryEmbedding,
    apply_rotary_pos_emb,
    should_use_fused_mla_rope,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant import (
    dsa_indexer_loss,
    dsa_kernels,
    dsa_layout,
    dsa_masking,
)
from megatron.core.transformer.experimental_attention_variant.dsa_layout import (
    build_packed_allgather_cp_local_positions_from_host as _build_cp_positions_from_host,
)
from megatron.core.transformer.experimental_attention_variant.dsa_layout import (
    build_packed_allgather_cp_query_positions_and_key_reorder_from_host as _cp_reorder_from_host,
)
from megatron.core.transformer.experimental_attention_variant.dsa_logging import (
    DSAIndexerLossLoggingHelper,
)
from megatron.core.transformer.forward_sharing import (
    CheckpointPhase,
    get_forward_sharing_state,
    is_mtp_repeated_sharing_source,
)
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import ensure_params_ready, get_pg_size

logger = logging.getLogger(__name__)
_DSA_WEIGHTS_PROJ_TE_GEMM_FALLBACK_WARNED = False
_DSA_CHECKPOINT_TOPK_REUSE_FALLBACK_WARNED = False
_warned_mtp_index_sharing_fused_bypass = False
# Kinds of prebuilt packed-CP layout mismatch already warned about (geometry fields
# and/or device that differed), so each kind is reported once per process.
_DSA_PREBUILT_LAYOUT_MISMATCH_WARNED = set()

try:
    from transformer_engine.pytorch.module.base import get_dummy_wgrad
except ImportError:
    get_dummy_wgrad = None

try:
    from megatron.core.fusions.fused_mla_yarn_rope_apply import fused_mla_rope_inplace
except ImportError:
    fused_mla_rope_inplace = None

try:
    from fast_hadamard_transform import hadamard_transform
except ImportError:
    hadamard_transform = None


def is_dsa_skip_topk_layer(layer_number: int, skip_topk_offset: int, topk_freq: int) -> bool:
    """Return whether a 1-indexed layer reuses a previous DSA top-k result."""
    if layer_number < 1:
        raise ValueError(f"layer_number must be 1-indexed and positive, got {layer_number}.")
    if skip_topk_offset < 0:
        raise ValueError(f"skip_topk_offset must be non-negative, got {skip_topk_offset}.")
    if topk_freq < 1:
        raise ValueError(f"topk_freq must be positive, got {topk_freq}.")
    # Layers are 1-indexed, so the default offset 0 must still start at layer 1.
    skip_topk_offset = max(skip_topk_offset, 1)
    return (max(layer_number - skip_topk_offset, 0) % topk_freq) != 0


def source_dsa_compute_layer(layer_number: int, skip_topk_offset: int, topk_freq: int) -> int:
    """Return the computing layer whose DSA top-k a skip layer reuses."""
    is_dsa_skip_topk_layer(layer_number, skip_topk_offset, topk_freq)
    skip_topk_offset = max(skip_topk_offset, 1)
    if layer_number <= skip_topk_offset:
        return layer_number
    return layer_number - ((layer_number - skip_topk_offset) % topk_freq)


def _unfused_absorbed_dsa_fn(
    query: torch.Tensor,
    key: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    v_channels: int,
    mask: Optional[torch.Tensor] = None,
    varlen_starts: Optional[torch.Tensor] = None,
    varlen_ends: Optional[torch.Tensor] = None,
    key_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Unfused absorbed-MLA attention: output stays [sq, b, np, v_channels]."""
    sq, b, np, hn = query.size()
    skv = key.size(0)
    assert key.size(2) == 1, "Absorbed DSA expects MQA key head dimension = 1"
    assert key.size(-1) >= v_channels, "key last dim must contain latent value channels"
    row_mask, varlen_starts, varlen_ends, key_positions = dsa_masking.prepare_sparse_mask_context(
        mask=mask,
        varlen_starts=varlen_starts,
        varlen_ends=varlen_ends,
        key_positions=key_positions,
        sq=sq,
        sk=skv,
        b=b,
        device=query.device,
    )

    # [sq,b,np,hn] -> [b,np,sq,hn]
    q = query.permute(1, 2, 0, 3)
    # [skv,b,1,hn] -> [b,1,hn,skv]
    k = key.permute(1, 2, 3, 0)
    attention_scores = torch.matmul(q.float(), k.float()) * softmax_scale

    # Sparse + causal/varlen validity mask.
    index_mask = torch.full((b, sq, skv), float("-inf"), device=attention_scores.device)
    dsa_masking.scatter_topk_into_index_mask(index_mask, topk_indices, seq_chunk_size=256)
    index_mask = dsa_masking.apply_sparse_validity_to_index_mask(
        index_mask,
        row_mask=row_mask,
        varlen_starts=varlen_starts,
        varlen_ends=varlen_ends,
        key_positions=key_positions,
    )

    attention_scores = attention_scores + index_mask.unsqueeze(1)
    valid_index_mask = torch.isfinite(index_mask)
    attention_scores = dsa_masking.masked_softmax(
        attention_scores.float(), valid_index_mask.unsqueeze(1).expand(b, np, sq, skv), dim=-1
    )

    # Latent value is the first v_channels slice of absorbed key cache.
    value = key[..., :v_channels].permute(1, 2, 0, 3)  # [b,1,skv,v]
    output = torch.matmul(attention_scores.to(value.dtype), value)  # [b,np,sq,v]
    return output.permute(2, 0, 1, 3).contiguous()


def _run_sparse_attention(
    *,
    absorbed_mla: bool,
    query: torch.Tensor,
    key: torch.Tensor,
    value: Optional[torch.Tensor],
    up_v_weight: Optional[torch.Tensor],
    topk_indices: torch.Tensor,
    softmax_scale: float,
    config: TransformerConfig,
    mask: Optional[torch.Tensor],
    varlen_starts: Optional[torch.Tensor],
    varlen_ends: Optional[torch.Tensor],
    key_positions: Optional[torch.Tensor],
    topk_length: Optional[torch.Tensor] = None,
    all_topk_rows_nonempty: bool = False,
    return_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Run sparse attention for absorbed and non-absorbed MLA paths.

    Returns the attention output. With ``return_lse=True`` it returns ``(output, lse)``
    instead, where ``lse`` is the non-differentiable FP32 ``[sq, b, np]``
    natural-log LSE of the fused sparse-attention kernel (layout in
    ``dsa_cudnn_kernels.run_fused_absorbed_sparse_attention``). Only the fused absorbed-MLA
    path returns it: when no fused backend runs the call, ``return_lse=True`` raises instead
    of falling back to the PyTorch path, which computes no LSE.
    """
    if return_lse and not (absorbed_mla and dsa_kernels.use_fused_dsa_kernels(config)):
        raise RuntimeError(
            "return_lse=True needs the fused absorbed-MLA sparse attention (absorbed MLA and "
            "a fused dsa_kernel_backend); the unfused DSA attention does not return the LSE."
        )
    if absorbed_mla:
        latent_v_channels = int(getattr(config, "kv_lora_rank", 0) or 0)
        if latent_v_channels <= 0:
            raise RuntimeError(
                "Invalid kv_lora_rank for absorbed-MLA DSAttention sparse attention."
            )
        if up_v_weight is None:
            raise RuntimeError(
                "Absorbed DSAttention requires up_v_weight for latent-to-value projection."
            )
        if value is not None:
            raise RuntimeError(
                "Absorbed DSAttention expects value=None (latent path). "
                "Received absorbed layout with explicit value tensor."
            )
        output = None
        lse = None
        if dsa_kernels.use_fused_dsa_kernels(config):
            output = dsa_kernels.run_fused_absorbed_sparse_attention(
                config,
                query,
                key,
                topk_indices,
                softmax_scale,
                latent_v_channels,
                topk_length=topk_length,
                all_topk_rows_nonempty=all_topk_rows_nonempty,
                # Passed only when requested, so the default call is unchanged.
                **({"return_lse": True} if return_lse else {}),
            )
            if return_lse:
                if output is None:
                    raise RuntimeError(
                        f"DSA kernel backend {config.dsa_kernel_backend!r} declined the fused "
                        "sparse attention, and return_lse=True cannot fall back to the "
                        "unfused path, which does not return the LSE."
                    )
                output, lse = output
        # Fused backends may decline unsupported shapes or layouts by returning
        # None, so keep the absorbed PyTorch path as the authoritative fallback.
        if output is None:
            output = _unfused_absorbed_dsa_fn(
                query,
                key,
                topk_indices,
                softmax_scale,
                latent_v_channels,
                mask=mask,
                varlen_starts=varlen_starts,
                varlen_ends=varlen_ends,
                key_positions=key_positions,
            )
        assert output is not None
        output = torch.einsum("sbhc,hdc->sbhd", output, up_v_weight).contiguous()
        output = output.view(output.size(0), output.size(1), -1)
        if return_lse:
            return output, lse
        return output

    return unfused_dsa_fn(
        query,
        key,
        value,
        topk_indices,
        softmax_scale,
        mask=mask,
        varlen_starts=varlen_starts,
        varlen_ends=varlen_ends,
        key_positions=key_positions,
    )


def _normalize_dsattention_output_rank(output: torch.Tensor, target_ndim: int) -> torch.Tensor:
    """Normalize DSAttention output rank to match caller hidden-state rank."""
    if target_ndim not in (2, 3):
        raise RuntimeError(f"DSAttention expected x.ndim in (2, 3), got {target_ndim}")

    if output.ndim == 4:
        output = output.reshape(output.size(0), output.size(1), -1)
    elif output.ndim not in (2, 3):
        raise RuntimeError(
            f"DSAttention produced unexpected output rank {output.ndim}; expected 2D/3D/4D."
        )

    if target_ndim == 3 and output.ndim == 2:
        output = output.unsqueeze(1)
    elif target_ndim == 2 and output.ndim == 3:
        if output.size(1) != 1:
            raise RuntimeError(
                "DSAttention cannot squeeze non-singleton batch dim for packed output: "
                f"shape={tuple(output.shape)}"
            )
        output = output.squeeze(1)

    if output.ndim != target_ndim:
        raise RuntimeError(
            "DSAttention output rank mismatch after normalization: "
            f"target_ndim={target_ndim}, output_shape={tuple(output.shape)}"
        )
    return output


def _validate_nonpacked_cp_uniform_length(
    sq: int,
    skv: int,
    cp_size: int,
    cp_group: Optional[torch.distributed.ProcessGroup],
    device: torch.device,
) -> None:
    """Validate the uniform-length precondition for non-packed allgather CP."""
    expected_skv = sq * cp_size
    if (
        cp_group is not None
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and get_pg_size(cp_group) == cp_size
    ):
        local_len = torch.tensor([sq], device=device, dtype=torch.int64)
        all_lens = [torch.empty_like(local_len) for _ in range(cp_size)]
        torch.distributed.all_gather(all_lens, local_len, group=cp_group)
        all_lens = torch.cat(all_lens)
        if not torch.all(all_lens == sq):
            raise RuntimeError(
                "Non-packed DSA allgather CP expects uniform per-rank sequence lengths; "
                f"got per-rank lengths {all_lens.tolist()}."
            )
        expected_skv = int(all_lens.sum().item())

    if skv != sq and skv != expected_skv:
        raise RuntimeError(
            "Non-packed DSA allgather CP expects uniform per-rank sequence lengths; "
            f"got local query length {sq} and key length {skv} for cp_size={cp_size}."
        )


class _PermuteRows(torch.autograd.Function):
    """``x.index_select(0, perm)`` for a bijective ``perm``, with a gather as its backward.

    Autograd's generic ``index_select`` backward is a zero fill plus an atomic
    ``index_add_``. When ``perm`` is a permutation, every input row receives exactly
    one gradient row, so the backward is just ``grad.index_select(0, inv_perm)``: one
    coalesced gather, no fill and no atomics. The values are identical except that
    a gathered ``-0.0`` stays ``-0.0``, where adding it onto the zero fill gives
    ``+0.0``.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, perm: torch.Tensor, inv_perm: torch.Tensor) -> torch.Tensor:
        """Gather the rows of ``x`` in ``perm`` order."""
        ctx.save_for_backward(inv_perm)
        return x.index_select(0, perm)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Scatter the gradient back to input order by gathering with ``inv_perm``."""
        (inv_perm,) = ctx.saved_tensors
        return grad_output.index_select(0, inv_perm), None, None


def _permute_rows(
    x: torch.Tensor, perm: torch.Tensor, inv_perm: Optional[torch.Tensor]
) -> torch.Tensor:
    """Return ``x.index_select(0, perm)``, with a gather backward when ``inv_perm`` is given.

    Without ``inv_perm`` or without a backward to record (no grad mode, or an input
    that does not require grad), this is the plain ``index_select``.
    """
    if inv_perm is None or not (torch.is_grad_enabled() and x.requires_grad):
        return x.index_select(0, perm)
    return _PermuteRows.apply(x, perm, inv_perm)


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Apply Hadamard rotation activation.
    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L424-L428

    Args:
        x: Input tensor (must be bfloat16).

    Returns:
        Rotated tensor.
    """
    assert (
        x.dtype == torch.bfloat16
    ), f"rotate_activation only support bf16 input, but got {x.dtype}"
    assert hadamard_transform is not None, "fast_hadamard_transform is not installed."
    hidden_size = x.size(-1)
    return hadamard_transform(x, scale=hidden_size**-0.5)


def compute_dsa_indexer_loss(
    index_scores: torch.Tensor,
    topk_indices: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    softmax_scale: float,
    loss_coeff: float,
    sparse_loss: bool,
    pg_collection: ProcessGroupCollection,
    mask: Optional[torch.Tensor] = None,
    varlen_starts: Optional[torch.Tensor] = None,
    varlen_ends: Optional[torch.Tensor] = None,
    key_positions: Optional[torch.Tensor] = None,
    query_valid_rows: Optional[torch.Tensor] = None,
    calculate_per_token_loss: bool = False,
    non_compressed_lse: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute KL divergence loss between index_scores and true attention_scores.

    This loss trains the indexer to predict which tokens are important by matching the distribution
    of true attention scores.

    Reference: Section 2.1 of
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf

    Args:
        index_scores: Scores predicted by indexer [batch, seqlen_q, seqlen_k].
        topk_indices: Top-k indices [batch, seqlen_q, index_topk].
        query: Query tensor [seqlen_q, batch, heads, dim].
        key: Key tensor [seqlen_k, batch, heads, dim].
        softmax_scale: Scale coefficient after q @ k^T.
        loss_coeff: Coefficient for the indexer KL divergence loss.
        sparse_loss: bool, whether to use sparse indexer loss. If True, only the topk
            indices will be used to compute the loss.
        pg_collection: Process group collection, must have TP process group.
        mask: Optional additive attention mask. Supports shape [sq, sk] or [b, sq, sk].
            Invalid positions should be -inf.
        varlen_starts: Optional row-wise key start bounds [sq] for packed THD.
        varlen_ends: Optional row-wise key end bounds [sq] for packed THD.
        key_positions: Optional global key positions [sk] for packed THD.
        non_compressed_lse: Optional detached FP32 log-sum-exp contribution
            [batch, heads, seqlen_q] from teacher keys that are intentionally
            omitted from ``key``. When provided, the selected ``key`` logits
            are normalized with this external mass before heads are summed.

    Returns:
        index_loss: KL divergence loss (scalar).
    """
    query, _ = dsa_layout.ensure_sbhd(query, "query")
    key, _ = dsa_layout.ensure_sbhd(key, "key")

    sq, b, np, hn = query.size()
    sk = key.size(0)
    query_valid_rows = dsa_masking.normalize_query_valid_rows(
        query_valid_rows, b=b, sq=sq, device=index_scores.device
    )

    varlen_starts, varlen_ends, key_positions = dsa_masking.normalize_varlen_bounds(
        mask=mask,
        varlen_starts=varlen_starts,
        varlen_ends=varlen_ends,
        key_positions=key_positions,
        sk=sk,
        device=index_scores.device,
    )

    # [sq, b, np, hn] -> [b, np, sq, hn] -> [b * np, sq, hn]
    query = query.permute(1, 2, 0, 3).reshape(b * np, sq, hn)
    # [sk, b, np, hn] -> [b, np, hn, sk] -> [b * np, hn, sk]
    key = key.permute(1, 2, 3, 0).reshape(b * np, hn, sk)
    # Compute attention scores [b * np, sq, sk]
    attention_scores = torch.bmm(query.float(), key.float()) * softmax_scale
    # Reshape to [b, np, sq, sk]
    attention_scores = attention_scores.reshape(b, np, sq, sk)
    if varlen_starts is not None:
        attention_scores = dsa_masking.apply_starts_ends_mask_to_scores(
            attention_scores, varlen_starts, varlen_ends, key_positions
        )
        index_scores = dsa_masking.apply_starts_ends_mask_to_scores(
            index_scores, varlen_starts, varlen_ends, key_positions
        )
        base_valid_mask = (
            dsa_masking.build_valid_mask_from_starts_ends(varlen_starts, varlen_ends, key_positions)
            .unsqueeze(0)
            .expand(b, sq, sk)
        )
    else:
        _, attn_score_mask, index_score_mask, base_valid_mask = dsa_masking.prepare_additive_mask(
            mask, sq=sq, sk=sk, b=b, device=attention_scores.device
        )
        # [b, np, sq, sk] + [1/b, 1, sq, sk] -> [b, np, sq, sk]
        attention_scores += attn_score_mask
        # [b, sq, sk] + [1/b, sq, sk] -> [b, sq, sk]
        index_scores += index_score_mask

    # index_mask [b, sq, sk]
    index_mask = torch.full(
        (b, sq, sk), float("-inf"), dtype=torch.float32, device=attention_scores.device
    )
    dsa_masking.scatter_topk_into_index_mask(index_mask, topk_indices, seq_chunk_size=256)

    if sparse_loss:
        # [b, np, sq, sk] + [b, 1, sq, sk] -> [b, np, sq, sk]
        attention_scores += index_mask.view(b, 1, sq, sk)
        # [b, sq, sk] + [b, sq, sk] -> [b, sq, sk]
        index_scores += index_mask
        index_valid_mask = base_valid_mask & (index_mask == 0)
    else:
        index_valid_mask = base_valid_mask
    attention_valid_mask = index_valid_mask if sparse_loss else base_valid_mask

    # [b, np, sq, sk] -> [b, np, sq, sk]
    attention_scores = _compute_indexer_teacher_probabilities(
        attention_scores, attention_valid_mask, non_compressed_lse=non_compressed_lse
    )
    # [b, sq, sk] -> [b, sq, sk]
    index_log_scores = dsa_masking.masked_log_softmax(
        index_scores.float(), index_valid_mask, dim=-1
    )

    # Sum attention scores across heads.
    # [batch, heads, seqlen_q, seqlen_k] -> [batch, seqlen_q, seqlen_k]
    attention_scores = attention_scores.sum(dim=1)
    if pg_collection.tp.size() > 1:
        # attention scores are scattered to TP ranks in head dimension.
        torch.distributed.all_reduce(attention_scores.contiguous(), group=pg_collection.tp)
    # The target is already non-negative because it is a sum of softmax probabilities.
    attention_scores = _normalize_indexer_teacher_target(attention_scores, non_compressed_lse)
    return dsa_indexer_loss.indexer_loss_from_target(
        attention_scores,
        index_log_scores,
        loss_coeff,
        query_valid_rows=query_valid_rows,
        calculate_per_token_loss=calculate_per_token_loss,
    )


def _compute_indexer_teacher_probabilities(
    attention_scores: torch.Tensor,
    attention_valid_mask: torch.Tensor,
    non_compressed_lse: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return selected-key teacher mass, optionally including omitted mass.

    ``non_compressed_lse`` is a sufficient statistic for teacher logits that
    must participate in the softmax denominator but must not appear in the
    compressed-key target returned by this helper. When the absolute compressed
    mass underflows FP32, all heads in a row receive the same log-domain shift;
    the returned weights remain proportional and the caller L1-normalizes them.
    """
    b, np, sq, sk = attention_scores.shape
    expanded_valid_mask = attention_valid_mask.unsqueeze(1).expand(b, np, sq, sk)
    if non_compressed_lse is None:
        return dsa_masking.masked_softmax(attention_scores.float(), expanded_valid_mask, dim=-1)

    expected_shape = (b, np, sq)
    if tuple(non_compressed_lse.shape) != expected_shape:
        raise ValueError(
            "non_compressed_lse must have shape [batch, heads, seqlen_q], "
            f"got {tuple(non_compressed_lse.shape)}, expected {expected_shape}"
        )
    if non_compressed_lse.device != attention_scores.device:
        raise ValueError(
            "non_compressed_lse and attention_scores must be on the same device, "
            f"got {non_compressed_lse.device} and {attention_scores.device}"
        )
    if non_compressed_lse.requires_grad:
        raise ValueError("non_compressed_lse must be detached")

    masked_scores = attention_scores.float().masked_fill(~expanded_valid_mask, float("-inf"))
    compressed_lse = torch.logsumexp(masked_scores, dim=-1)
    row_has_compressed_keys = expanded_valid_mask.any(dim=-1)
    # Avoid the undefined ``-inf - -inf`` intermediate on fully masked rows.
    # This is only a [batch, heads, seqlen] tensor, so it does not recreate the
    # full-size temporary that the log-domain formulation is designed to avoid.
    safe_compressed_lse = torch.where(
        row_has_compressed_keys, compressed_lse, torch.zeros_like(compressed_lse)
    )
    conditional_probabilities = torch.exp(masked_scores - safe_compressed_lse.unsqueeze(-1))
    del masked_scores

    full_lse = torch.logaddexp(non_compressed_lse.float(), compressed_lse)
    log_compressed_mass = (compressed_lse - full_lse).masked_fill(
        ~row_has_compressed_keys, float("-inf")
    )

    # The external window/sink mass can put every head's compressed mass below
    # the FP32 normal range. A common per-row shift across heads
    # preserves all relative teacher weights and cancels in the downstream L1
    # normalization. CSA currently requires TP1, so no cross-rank MAX is needed.
    row_max = log_compressed_mass.amax(dim=1, keepdim=True)
    needs_rescale = torch.isfinite(row_max) & (row_max < math.log(torch.finfo(torch.float32).tiny))
    common_shift = torch.where(needs_rescale, row_max, torch.zeros_like(row_max))
    compressed_mass = torch.exp(log_compressed_mass - common_shift)
    return conditional_probabilities * compressed_mass.unsqueeze(-1)


def _normalize_indexer_teacher_target(
    target: torch.Tensor, non_compressed_lse: torch.Tensor | None
) -> torch.Tensor:
    """L1-normalize teacher mass without changing the legacy DSA path."""
    if non_compressed_lse is None:
        return dsa_indexer_loss.normalize_indexer_target(target)
    return _normalize_teacher_target_exact_zero(target)


def _normalize_teacher_target_exact_zero(target: torch.Tensor) -> torch.Tensor:
    """L1-normalize non-negative teacher mass, keeping only exactly zero rows at zero."""
    row_mass = target.sum(dim=-1, keepdim=True)
    # External teacher mass can legitimately make the compressed mass smaller
    # than INDEXER_LOSS_EPS (or even float32 tiny). Only an exactly zero row is
    # degenerate; keep it zero rather than imposing a numerical floor.
    safe_row_mass = torch.where(row_mass > 0, row_mass, torch.ones_like(row_mass))
    return target / safe_row_mass


def _compute_index_scores(
    q: torch.Tensor, weights: torch.Tensor, k: torch.Tensor, use_relu: bool = True
) -> torch.Tensor:
    """
    Perform index scoring with BF16/FP32 inputs.

    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/kernel.py#L254-L274
    This implementation accepts BF16/FP32 inputs for the `fp8_index` scoring logic:
        1. Compute attention scores: q @ k^T;
        2. Optionally apply ReLU activation (DeepSeek V3.2 only; disabled for GLM5);
        3. Weight by attention weights;
        4. Sum across attention heads.

    Args:
        q: BF16/FP32 [seqlen_q, batch, index_n_heads, index_head_dim], the query tensor.
        weights: BF16/FP32 [seqlen_q, batch, index_n_heads], the attention weights.
        k: BF16/FP32 [seqlen_k, batch, index_head_dim], the key tensor.

    Returns:
        index_scores: FP32 [batch, seqlen_q, seqlen_k], the index scores.
    """
    sq, batch, n_heads, head_dim = q.shape
    sk = k.size(0)
    if sq == 0:
        return torch.empty(batch, 0, sk, dtype=torch.float32, device=q.device)
    k_fp32 = k.float()

    # Chunk over seqlen_q to avoid materializing the full [sq, batch, heads, sk]
    # fp32 tensor. Target 256 MiB per chunk to leave room for the output.
    bytes_per_token = batch * n_heads * sk * 4
    chunk_size = min(sq, max(1, 256 * 1024 * 1024 // max(1, bytes_per_token)))
    needs_grad = torch.is_grad_enabled() and any(tensor.requires_grad for tensor in (q, weights, k))
    if needs_grad:
        score_chunks = []
    else:
        index_scores = torch.empty(sq, batch, sk, dtype=torch.float32, device=q.device)

    for start in range(0, sq, chunk_size):
        end = min(start + chunk_size, sq)
        # [chunk, batch, heads, sk]
        scores = torch.einsum('sbhd,tbd->sbht', q[start:end].float(), k_fp32)
        if use_relu:
            scores.relu_()
        # Top-k selection runs without gradients and can reuse the score buffer. When autograd
        # is active, collect chunks so the returned scores retain their computation graph.
        head_weights = weights[start:end].unsqueeze(-1)
        if needs_grad:
            scores = scores * head_weights
        else:
            scores.mul_(head_weights)
        score_chunk = scores.sum(dim=2)
        if needs_grad:
            score_chunks.append(score_chunk)
        else:
            index_scores[start:end] = score_chunk
        del scores

    # Transpose to [batch, seqlen_q, seqlen_k].
    if needs_grad:
        index_scores = torch.cat(score_chunks, dim=0)
    index_scores = index_scores.transpose(0, 1)

    return index_scores


def fused_qk_topk_naive(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    index_topk: int,
    mask: Optional[torch.Tensor] = None,
    varlen_starts: Optional[torch.Tensor] = None,
    varlen_ends: Optional[torch.Tensor] = None,
    key_positions: Optional[torch.Tensor] = None,
    use_relu: bool = True,
):
    """Naive implementation of QK Topk."""
    sk = k.size(0)
    # =========================================
    # Compute index scores
    # =========================================
    # [batch, seqlen, seqlen]
    index_scores = _compute_index_scores(q, weights, k, use_relu=use_relu)
    varlen_starts, varlen_ends, key_positions = dsa_masking.normalize_varlen_bounds(
        mask=mask,
        varlen_starts=varlen_starts,
        varlen_ends=varlen_ends,
        key_positions=key_positions,
        sk=sk,
        device=index_scores.device,
    )
    if varlen_starts is not None:
        index_scores = dsa_masking.apply_starts_ends_mask_to_scores(
            index_scores, varlen_starts, varlen_ends, key_positions
        )
    elif mask is not None:
        assert mask.dtype == index_scores.dtype, "Mask dtype must match index scores dtype"
        index_scores = index_scores + mask

    # =========================================
    # Select top-k indices
    # =========================================
    topk_k = min(index_topk, sk)
    if topk_k > 0:
        topk_scores, topk_indices = index_scores.topk(topk_k, dim=-1)
        topk_indices = topk_indices.masked_fill(topk_scores == float("-inf"), -1)
    else:
        topk_indices = torch.empty(
            index_scores.shape[:-1] + (0,), dtype=torch.int64, device=index_scores.device
        )

    return index_scores, topk_indices


def _kpool_fp8_input(x: torch.Tensor) -> torch.Tensor:
    """Match the indexer's FP32 Hadamard, BF16 rounding, and E4M3 power-of-two scale."""
    if not x.numel():
        return x.float()
    assert hadamard_transform is not None, "fast_hadamard_transform is required for FP8 KPool."
    x = hadamard_transform(x.float(), scale=x.shape[-1] ** -0.5).to(torch.bfloat16).float()
    absmax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(absmax / 448.0)))
    return (x / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale


def _kpool_compress_keys(
    k: torch.Tensor,
    gate_score: torch.Tensor,
    ape: torch.Tensor,
    pool_size: int,
    kpool_key_dtype_compression: bool = False,
) -> torch.Tensor:
    """Softmax-weighted pool keys with an explicit compression rounding profile.

    Keys and gates are [tokens, batch, head_dim]; ape is [pool_size, head_dim].
    Only complete pools are compressed. Query-local tails are appended separately.
    The default preserves the FP32 reduction and BF16 output of the existing KPool
    implementation (#8287). The key-dtype profile rounds probabilities and products to the
    key dtype before the reduction, as the Hugging Face reference KPool indexer does
    (transformers v5.16.1, Glm5NextTextIndexer.get_pooled_states).
    """
    seqlen, bsz, head_dim = k.shape
    assert head_dim == ape.shape[1], f"head_dim {head_dim} != ape dim1 {ape.shape[1]}"
    num_pools = seqlen // pool_size
    # Drop the trailing incomplete pool from compression; its tokens are appended
    # later via append_tail_to_topk (always_select_tail).
    usable = num_pools * pool_size
    # [num_pools, pool_size, batch, head_dim]
    k_p = k[:usable].reshape(num_pools, pool_size, bsz, head_dim)
    # gate_score: [seqlen, batch, head_dim] -> [num_pools, pool_size, batch, head_dim]
    if gate_score is not None:
        g = gate_score[:usable].reshape(num_pools, pool_size, bsz, head_dim).float()
    else:
        g = torch.zeros((num_pools, pool_size, bsz, head_dim), dtype=torch.float32, device=k.device)

    # Per-dim softmax across the pool's slots: score[slot] = gate_score[slot] + ape[slot].
    # ape: [pool_size, head_dim] -> broadcast over (num_pools, batch).
    ape_f = ape.to(dtype=torch.float32, device=k.device)  # [pool_size, head_dim]
    score = g + ape_f.unsqueeze(0).unsqueeze(2)  # [num_pools, pool_size, batch, head_dim]
    if kpool_key_dtype_compression:
        # The Hugging Face reference (transformers v5.16.1, Glm5NextTextIndexer.
        # get_pooled_states) casts the FP32 softmax to the key dtype before
        # multiplying, not after an FP32 weighted sum.
        probabilities = torch.nan_to_num(score.softmax(dim=1)).to(k_p.dtype)
        return (probabilities * k_p).sum(dim=1)
    # Numerically-stable per-dim softmax over dim=1 (the pool slot dim). The shift cancels
    # in the ratio below, so its exact gradient is zero; detach it so a trainable KPool
    # indexer does not route a (rounding-only) gradient through the max.
    score_max = score.detach().max(dim=1, keepdim=True).values
    prob = torch.exp(score - score_max)
    # weighted sum of k over pool slots: [num_pools, batch, head_dim]. Keep the
    # numerator 4D ([num_pools, 1, batch, head_dim]) so it broadcasts cleanly
    # against the 4D denom ([num_pools, 1, 1, head_dim]); a 3D numerator would
    # left-pad and produce a spurious extra (num_pools) dimension.
    k_f = k_p.float()
    k_pooled = (prob * k_f).sum(dim=1, keepdim=True) / prob.sum(dim=1, keepdim=True).clamp(
        min=1e-12
    )
    k_pooled = k_pooled.squeeze(1)  # [num_pools, batch, head_dim]
    return k_pooled.to(torch.bfloat16)


def _expand_pools_to_tokens(
    pool_ids: torch.Tensor,
    pool_valid: torch.Tensor,
    topk_tokens: int,
    pool_size: int,
    pool_token_base: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Expand fixed-width pool IDs to token IDs, preserving -1 padding."""
    assert pool_ids.ndim == 2 and pool_valid.shape == pool_ids.shape
    assert pool_ids.shape[1] * pool_size == topk_tokens
    if pool_token_base is None:
        starts = pool_ids * pool_size
    elif pool_token_base.numel():
        starts = pool_token_base[pool_ids.clamp(min=0)]
    else:
        starts = torch.zeros_like(pool_ids)
    offsets = torch.arange(pool_size, device=pool_ids.device)
    tokens = starts.unsqueeze(-1) + offsets
    tokens = tokens.masked_fill(~pool_valid.unsqueeze(-1), -1)
    return tokens.reshape(pool_ids.shape[0], topk_tokens).to(torch.int32)


def _append_tail_to_topk(
    topk_result: torch.Tensor,
    seq_lens: torch.Tensor,
    pool_size: int,
    tail_start_override: Optional[torch.Tensor] = None,
    tail_count_override: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Append each query's incomplete causal pool, in global token coordinates."""
    tail_count = (
        seq_lens.to(torch.int32).remainder(pool_size)
        if tail_count_override is None
        else tail_count_override.to(torch.int32)
    )
    tail_start = (
        seq_lens.to(torch.int32) - tail_count
        if tail_start_override is None
        else tail_start_override.to(torch.int32)
    )
    offsets = torch.arange(pool_size - 1, device=topk_result.device)
    tail = tail_start[:, None] + offsets
    tail = tail.masked_fill(offsets >= tail_count[:, None], -1).to(topk_result.dtype)
    return torch.cat((topk_result, tail), dim=-1)


def _fill_kpool_left_boundary(
    token_topk: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor, pool_size: int
) -> torch.Tensor:
    """Use unused expanded-pool slots for a left partial pool from an explicit mask."""
    rows, width = token_topk.shape
    complete_start = ((starts + pool_size - 1) // pool_size) * pool_size
    complete_end = (ends // pool_size) * pool_size
    has_complete_pool = complete_start < complete_end
    left_end = torch.where(starts < ends, torch.minimum(complete_start, ends), starts)
    left_count = (left_end - starts).clamp_min(0)
    left_count = torch.minimum(
        left_count,
        torch.where(
            has_complete_pool,
            torch.full_like(left_count, pool_size - 1),
            torch.full_like(left_count, pool_size),
        ),
    )

    offsets = torch.arange(pool_size, device=token_topk.device, dtype=torch.int32)
    left_tokens = starts[:, None].to(torch.int32) + offsets
    left_valid = offsets[None, :] < left_count[:, None]

    invalid = token_topk < 0
    invalid_positions = torch.where(
        invalid,
        torch.arange(width, device=token_topk.device, dtype=torch.int32)[None, :],
        torch.full((rows, width), width, device=token_topk.device, dtype=torch.int32),
    )
    fill_positions = invalid_positions.topk(k=min(pool_size, width), dim=-1, largest=False).values
    fill_positions_safe = fill_positions.clamp_max(width - 1)
    fill_valid = left_valid[:, : fill_positions.size(1)] & (fill_positions < width)
    current = token_topk.gather(-1, fill_positions_safe)
    values = torch.where(fill_valid, left_tokens[:, : fill_positions.size(1)], current)
    return token_topk.scatter(-1, fill_positions_safe, values.to(token_topk.dtype))


def _kpool_compress_keys_per_seg(
    k: torch.Tensor,
    gate_score: Optional[torch.Tensor],
    ape: torch.Tensor,
    pool_size: int,
    cu_seqlens_kv: torch.Tensor,
    kpool_key_dtype_compression: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compress complete pools within each packed segment.

    Return pooled keys and their global starting token indices. Segment boundaries
    need not be multiples of pool_size; a pool must never span two documents.
    """
    cu = cu_seqlens_kv.to(device=k.device, dtype=torch.int64)
    sk, bsz, head_dim = k.shape
    n_seg = cu.numel() - 1
    max_pools = sk // pool_size
    if max_pools == 0:
        return (
            k.new_empty(
                (0, bsz, head_dim), dtype=k.dtype if kpool_key_dtype_compression else torch.bfloat16
            ),
            torch.empty(0, device=k.device, dtype=torch.int64),
        )

    # Enumerate a fixed upper bound of complete pools and map each pool to its
    # segment. Invalid slots stay allocated and are excluded from top-k below.
    pools_per_segment = torch.div(cu[1:] - cu[:-1], pool_size, rounding_mode="floor")
    pool_offsets = torch.cat(
        (torch.zeros(1, device=k.device, dtype=torch.int64), pools_per_segment.cumsum(0))
    )
    pool_ids = torch.arange(max_pools, device=k.device, dtype=torch.int64)
    segment_ids = torch.searchsorted(pool_offsets[1:], pool_ids, right=True).clamp_max(n_seg - 1)
    pool_token_base = cu[segment_ids] + (pool_ids - pool_offsets[segment_ids]) * pool_size
    pool_valid = pool_ids < pool_offsets[-1]
    token_positions = pool_token_base.unsqueeze(-1) + torch.arange(pool_size, device=k.device)
    token_positions = token_positions.clamp(0, sk - 1).reshape(-1)
    pooled_gate = gate_score[token_positions] if gate_score is not None else None
    k_pooled = _kpool_compress_keys(
        k[token_positions],
        pooled_gate,
        ape,
        pool_size,
        kpool_key_dtype_compression=kpool_key_dtype_compression,
    )
    pool_token_base = pool_token_base.masked_fill(~pool_valid, -1)
    return k_pooled, pool_token_base


def _pool_validity_from_token_mask(
    mask: torch.Tensor, pool_token_base: torch.Tensor, pool_size: int, sk: int
) -> torch.Tensor:
    """Map token validity to complete-pool validity for KPool scoring."""
    if mask.ndim not in (2, 3) or mask.shape[-1] != sk:
        raise ValueError(
            "KPool attention mask must have shape [sq, sk] or [batch, sq, sk], "
            f"got {tuple(mask.shape)} for sk={sk}"
        )
    token_positions = pool_token_base.clamp_min(0).unsqueeze(-1) + torch.arange(
        pool_size, device=mask.device
    )
    token_positions = token_positions.clamp_max(sk - 1)
    if mask.ndim == 2:
        token_valid = torch.gather(
            torch.isfinite(mask).unsqueeze(1).expand(-1, pool_token_base.numel(), -1),
            -1,
            token_positions.unsqueeze(0).expand(mask.size(0), -1, -1),
        )
    else:
        token_valid = torch.gather(
            torch.isfinite(mask).unsqueeze(-2).expand(-1, -1, pool_token_base.numel(), -1),
            -1,
            token_positions.view(1, 1, -1, pool_size).expand(mask.size(0), mask.size(1), -1, -1),
        )
    pool_valid = token_valid.all(dim=-1)
    return pool_valid & (pool_token_base >= 0)


def _query_bounds_from_token_mask(mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return global valid key bounds for each query row of an explicit mask."""
    valid = torch.isfinite(mask)
    sk = mask.size(-1)
    key_positions = torch.arange(sk, device=mask.device, dtype=torch.int64)
    starts = key_positions.masked_fill(~valid, sk).amin(dim=-1)
    ends = key_positions.masked_fill(~valid, -1).amax(dim=-1) + 1
    has_valid = valid.any(dim=-1)
    starts = starts.masked_fill(~has_valid, 0)
    ends = ends.masked_fill(~has_valid, 0)
    return starts, ends


def _expand_query_bounds_for_batch(bounds: torch.Tensor, batch: int, seqlen: int) -> torch.Tensor:
    """Normalize explicit-mask bounds to [batch, seqlen] without materializing masks."""
    if bounds.ndim == 1:
        if bounds.numel() != seqlen:
            raise ValueError(f"Expected {seqlen} query bounds, got {bounds.numel()}.")
        return bounds.unsqueeze(0).expand(batch, -1)
    if bounds.ndim == 2 and bounds.shape == (batch, seqlen):
        return bounds
    raise ValueError(
        f"Expected query bounds with shape [{seqlen}] or [{batch}, {seqlen}], "
        f"got {tuple(bounds.shape)}."
    )


def _kpool_explicit_mask_pool_phase(mask: torch.Tensor, pool_size: int) -> torch.Tensor:
    """Return the pool-grid phase of each sample's first valid key in an explicit mask.

    A key that no query row can see is padding, so the first valid key is the first key
    visible to any row: one value for a shared [sq, sk] mask, or [batch] for a
    [batch, sq, sk] mask. The Hugging Face reference KPool indexer starts pools at that
    key; earlier pools are invalid, so only its offset modulo pool_size matters. Samples
    without a visible key get phase 0. Computed on device without a host sync.
    """
    sk = mask.size(-1)
    key_visible = torch.isfinite(mask).any(dim=-2)
    positions = torch.arange(sk, device=mask.device, dtype=torch.int64).expand_as(key_visible)
    first_key = positions.masked_fill(~key_visible, sk).amin(dim=-1)
    return first_key.masked_fill(first_key == sk, 0).remainder(pool_size)


def _shift_kpool_explicit_mask_inputs(
    k: torch.Tensor, gate_score: Optional[torch.Tensor], mask: torch.Tensor, phase: torch.Tensor
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Shift each sample's key axis left by its pool phase: x'[t] = x[t + phase].

    Keys and gates are [keys, batch, dim]; the mask is [sq, sk] or [batch, sq, sk] and
    phase matches its leading shape. Positions past the last key get zero keys and
    gates and are masked, so they only belong to invalid pools. A zero phase is an
    exact identity.
    """
    sk, batch = k.shape[:2]
    positions = torch.arange(sk, device=k.device, dtype=torch.int64)
    source = positions + phase.unsqueeze(-1)
    in_range = source < sk
    source = source.clamp_max(sk - 1)
    shifted_mask = torch.gather(mask, -1, source.unsqueeze(-2).expand_as(mask))
    shifted_mask = shifted_mask.masked_fill(~in_range.unsqueeze(-2), float("-inf"))

    key_source = source.reshape(-1, sk).t().expand(sk, batch).unsqueeze(-1)
    key_in_range = in_range.reshape(-1, sk).t().expand(sk, batch).unsqueeze(-1)

    def shift(x: torch.Tensor) -> torch.Tensor:
        shifted = torch.gather(x, 0, key_source.expand(-1, -1, x.size(-1)))
        return shifted.masked_fill(~key_in_range, 0)

    return shift(k), shift(gate_score) if gate_score is not None else None, shifted_mask


_KPOOL_SCORE_CHUNK_BYTES = 2 * 1024 * 1024 * 1024


def _compute_kpool_hf_scores(
    q: torch.Tensor, k_pooled: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Hugging Face reference KPool scores: FP32 scaling, ReLU and head-reduction GEMM.

    Follows Glm5NextTextIndexer.forward in Hugging Face transformers v5.16.1.
    ``weights`` contains only the head-count scale, not the QK head-dimension
    scale. Preserve the reference operation order rather than moving that
    scale through ReLU or replacing the head-reduction GEMM with mul/sum.
    """
    with torch.autocast(device_type=q.device.type, enabled=False):
        queries = q.permute(1, 0, 2, 3).float()  # [batch, query, heads, dim]
        keys = k_pooled.permute(1, 2, 0).float().unsqueeze(1)
        scores = torch.relu(torch.matmul(queries, keys) * (q.size(-1) ** -0.5))
        head_weights = weights.permute(1, 0, 2).float().unsqueeze(-2)
        return torch.matmul(head_weights, scores).squeeze(-2)


@torch.no_grad()
def _kpool_topk_from_score_chunks(
    q: torch.Tensor,
    k_pooled: torch.Tensor,
    weights: torch.Tensor,
    index_topk: int,
    pool_size: int,
    pool_exists: Optional[torch.Tensor],
    mask: Optional[torch.Tensor],
    varlen_starts: Optional[torch.Tensor],
    varlen_ends: Optional[torch.Tensor],
    key_positions: Optional[torch.Tensor],
    pool_token_base: torch.Tensor,
    sk: int,
    use_relu: bool,
    hf_scoring: bool = False,
) -> torch.Tensor:
    """Compute KPool scores and selection in query chunks.

    KPool selection is discrete and runs under ``no_grad``. Only the fixed-width pool IDs
    are retained, not the full ``[batch, query, pool]`` score tensor.
    """
    sq, batch, n_heads, _ = q.shape
    num_pools = k_pooled.size(0)
    budget = index_topk // pool_size
    pool_topk = torch.full((batch, sq, budget), -1, dtype=torch.int64, device=q.device)
    if sq == 0 or budget == 0 or num_pools == 0:
        return pool_topk

    bytes_per_token = batch * n_heads * num_pools * 4
    chunk_size = min(sq, max(1, _KPOOL_SCORE_CHUNK_BYTES // max(1, bytes_per_token)))
    select_k = min(budget, num_pools)
    k_fp32 = k_pooled.float()
    pool_exists_mask = pool_exists.view(1, 1, -1) if pool_exists is not None else None

    for start in range(0, sq, chunk_size):
        end = min(start + chunk_size, sq)
        if hf_scoring:
            score_chunk = _compute_kpool_hf_scores(q[start:end], k_pooled, weights[start:end])
        else:
            scores = torch.einsum('sbhd,tbd->sbht', q[start:end].float(), k_fp32)
            if use_relu:
                scores.relu_()
            scores.mul_(weights[start:end].unsqueeze(-1))
            score_chunk = scores.sum(dim=2).transpose(0, 1)
            del scores

        if varlen_starts is not None:
            valid = dsa_masking.build_valid_mask_from_starts_ends(
                varlen_starts[start:end], varlen_ends[start:end], key_positions
            )
            score_chunk.masked_fill_(~valid.unsqueeze(0), float('-inf'))
        elif mask is not None:
            mask_chunk = mask[start:end] if mask.ndim == 2 else mask[:, start:end]
            pool_valid = _pool_validity_from_token_mask(mask_chunk, pool_token_base, pool_size, sk)
            score_chunk.masked_fill_(~pool_valid, float('-inf'))
        if pool_exists_mask is not None:
            score_chunk.masked_fill_(~pool_exists_mask, float('-inf'))

        topk_scores, topk_indices = score_chunk.topk(select_k, dim=-1)
        topk_indices.masked_fill_(topk_scores == float('-inf'), -1)
        pool_topk[:, start:end, :select_k] = topk_indices

    return pool_topk


def _mask_topk_tokens_with_token_mask(
    topk_indices: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Remove expanded KPool tokens that are invalid in an explicit token mask."""
    token_valid = torch.isfinite(mask)
    if token_valid.ndim == 2:
        token_valid = token_valid.unsqueeze(0)
    if token_valid.size(0) == 1 and topk_indices.size(0) != 1:
        token_valid = token_valid.expand(topk_indices.size(0), -1, -1)
    token_ids = topk_indices.clamp_min(0).to(torch.long)
    selected_valid = torch.gather(token_valid, -1, token_ids)
    return topk_indices.masked_fill((topk_indices < 0) | ~selected_valid, -1)


def fused_qk_topk_kpool(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    index_topk: int,
    pool_size: int,
    gate_score: torch.Tensor,
    ape: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    varlen_starts: Optional[torch.Tensor] = None,
    varlen_ends: Optional[torch.Tensor] = None,
    key_positions: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    use_relu: bool = True,
    always_select_tail: bool = True,
    fp8_indexer: bool = False,
    rotate_activation_enabled: bool = False,
    return_index_scores: bool = True,
    kpool_key_dtype_compression: bool = False,
    hf_scoring: bool = False,
    pool_topk_fn: Optional[Callable[..., Optional[torch.Tensor]]] = None,
    pooled_keys: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    return_selection: bool = False,
):
    """Select complete causal pools and append each query's incomplete tail.

    q is [queries, batch, heads, dim], k/gate_score are [keys, batch, dim],
    and weights are [queries, batch, heads]. Packed bounds use global token
    coordinates. Output indices are [batch, queries, index_topk + pool_size - 1]
    when always_select_tail is enabled, with -1 for unused slots.
    Set return_index_scores=False to select in query chunks without retaining scores.
    Compression rounding is independent of Hadamard rotation and score quantization.
    With hf_scoring, weights must contain only the FP32 head-count scale.

    ``pool_topk_fn``, when given, selects the pools of an unmasked, unpacked call without
    retained scores instead of the PyTorch chunk loop:
    ``pool_topk_fn(q, k_pooled, weights, pool_topk, pool_size, hf_scoring)`` receives the
    scoring inputs after any rotation or FP8 rounding and returns ``[batch, queries,
    pool_topk]`` pool ids with ``-1`` for unused slots, or None to keep the PyTorch loop.
    The caller owns the causal rule it applies (row at position ``p`` selects among pools
    ``[0, (p + 1) // pool_size)``, as ``varlen_starts``/``varlen_ends`` here).

    Without packing, an explicit mask starts the pool grid at each sample's first
    valid key, as the Hugging Face reference KPool indexer (transformers v5.16.1,
    Glm5NextTextIndexer) does. That key is the first one visible to any query
    row of the sample's [batch, sq, sk] mask, or of a shared [sq, sk] mask; earlier
    keys are padding. Index scores use the same grid: pool c covers keys from
    phase + c * pool_size, where phase is the first valid key modulo pool_size.
    Explicit-mask support:

    - Contiguous left and/or right padding with causal rows: HF parity. Padded query
      rows must be fully masked. A mask that lets padded rows see every key hides
      the padding, so pools keep the global grid.
    - Masks with gaps, sliding windows or row-dependent starts: tokens stay inside
      the mask, but leading partial pools only use unused slots and there is no HF
      parity claim.
    - Packed THD (cu_seqlens_kv) restarts pools per document; no mask and varlen
      bounds use the global grid. Neither is shifted.

    Model-level left padding is unsupported: causal DSA layers ignore the attention
    mask and treat padded tokens as real keys.

    ``pooled_keys``, when given, is a precomputed ``(k_pooled, pool_token_base)``: the
    pre-rotation pooled keys ``[pools, batch, dim]`` and each pool's first token in the
    key coordinates of ``varlen_starts``/``varlen_ends`` (``-1`` for an invalid packed
    capacity pool), as ``_kpool_compress_keys`` or ``_kpool_compress_keys_per_seg`` with
    ``cu_seqlens_kv`` returns them. Compression is skipped, so ``k``, ``gate_score`` and
    ``ape`` are not read; rotation and selection run as without it. An explicit ``mask``
    is rejected, because its pool phase shifts the keys before compression. With
    ``return_selection``, a third value is returned: the selected pool ids ``[batch,
    queries, index_topk // pool_size]`` (int64, ``-1`` for unused slots) in the
    selector's slot order, so that pool slot ``j`` expands to token slots
    ``j * pool_size .. j * pool_size + pool_size - 1`` (with an explicit mask, the ids
    index the phase-shifted pool grid and the mask may refill or drop token slots).
    Callers that need no autograd graph run this function under ``no_grad``; the
    selection itself is discrete.
    """
    if hf_scoring and not use_relu:
        raise ValueError("HF KPool scoring requires use_relu=True.")
    if pooled_keys is not None and mask is not None:
        raise ValueError(
            "fused_qk_topk_kpool cannot take precomputed pooled_keys with an explicit "
            "attention mask: the mask's pool phase shifts the keys before compression."
        )
    sk = k.size(0)
    if q.size(0) == 0:
        # Empty packed-CP ranks still need a fixed-width index tensor; reshape
        # with an inferred dimension is ambiguous when there are no query rows.
        width = index_topk + (pool_size - 1 if always_select_tail else 0)
        scores = q.new_empty((q.size(1), 0, sk // pool_size), dtype=torch.float32)
        indices = torch.empty((q.size(1), 0, width), dtype=torch.int32, device=q.device)
        if return_selection:
            selected_pools = torch.empty(
                (q.size(1), 0, index_topk // pool_size), dtype=torch.int64, device=q.device
            )
            return scores if return_index_scores else None, indices, selected_pools
        return scores if return_index_scores else None, indices

    # Packed pools restart at each document boundary.
    use_per_seg = cu_seqlens_kv is not None and cu_seqlens_kv.numel() >= 2
    # Explicit masks pool from each sample's first valid key. Select on a key axis
    # shifted by its pool phase, then map tokens back and mask with the caller's mask.
    token_mask = mask
    pool_phase = None
    explicit_mask = mask is not None and mask.ndim in (2, 3) and mask.size(-1) == sk
    if explicit_mask and not use_per_seg and sk > 0:
        pool_phase = _kpool_explicit_mask_pool_phase(mask, pool_size)
        k, gate_score, mask = _shift_kpool_explicit_mask_inputs(k, gate_score, mask, pool_phase)
    if pooled_keys is not None:
        k_pooled, pool_token_base = pooled_keys
        num_pools = k_pooled.size(0)
        if pool_token_base.numel() != num_pools:
            raise ValueError(
                "pooled_keys must hold one pool_token_base entry per pooled key, got "
                f"{pool_token_base.numel()} entries for {num_pools} pooled keys."
            )
        pool_token_base = pool_token_base.to(device=k_pooled.device, dtype=torch.int64)
    elif use_per_seg:
        k_pooled, pool_token_base = _kpool_compress_keys_per_seg(
            k,
            gate_score,
            ape,
            pool_size,
            cu_seqlens_kv,
            kpool_key_dtype_compression=kpool_key_dtype_compression,
        )
        num_pools = k_pooled.size(0)
    else:
        num_pools = sk // pool_size
        k_pooled = _kpool_compress_keys(
            k, gate_score, ape, pool_size, kpool_key_dtype_compression=kpool_key_dtype_compression
        )
        pool_token_base = torch.arange(num_pools, device=k.device, dtype=torch.int64) * pool_size

    if fp8_indexer:
        q, k_pooled = _kpool_fp8_input(q), _kpool_fp8_input(k_pooled)
    elif rotate_activation_enabled:
        q = rotate_activation(q)
        if k_pooled.numel():
            k_pooled = rotate_activation(k_pooled)
    # A pool is causal only when its final token is within the query's bounds.
    pool_exists = pool_token_base >= 0
    pool_positions = pool_token_base.clamp_min(0) + (pool_size - 1)
    eff_key_positions = (
        key_positions[pool_positions] if key_positions is not None else pool_positions
    )
    v_starts, v_ends, k_pos = dsa_masking.normalize_varlen_bounds(
        mask=mask,
        varlen_starts=varlen_starts,
        varlen_ends=varlen_ends,
        key_positions=eff_key_positions,
        sk=num_pools,
        device=q.device,
    )
    if return_index_scores:
        if hf_scoring:
            index_scores = _compute_kpool_hf_scores(q, k_pooled, weights)
        else:
            index_scores = _compute_index_scores(q, weights, k_pooled, use_relu=use_relu)
        if v_starts is not None:
            index_scores = dsa_masking.apply_starts_ends_mask_to_scores(
                index_scores, v_starts, v_ends, k_pos
            )
        elif mask is not None:
            assert mask.dtype == index_scores.dtype, "Mask dtype must match index scores dtype"
            pool_exists = pool_exists & _pool_validity_from_token_mask(
                mask, pool_token_base, pool_size, sk
            )
        index_scores = index_scores.masked_fill(~pool_exists, float("-inf"))

        # Keep the selection width fixed, including when fewer causal pools exist.
        budget = index_topk // pool_size
        select_k = min(budget, num_pools)
        if select_k > 0:
            topk_scores, pool_topk = index_scores.topk(select_k, dim=-1)
            # [batch, seqlen_q, select_k] -> mask invalid pools
            pool_topk = pool_topk.masked_fill(topk_scores == float("-inf"), -1)
        else:
            pool_topk = torch.empty(
                index_scores.shape[:-1] + (0,), dtype=torch.int64, device=index_scores.device
            )
        if pool_topk.shape[-1] < budget:
            pad = torch.full(
                index_scores.shape[:-1] + (budget - pool_topk.shape[-1],),
                -1,
                dtype=torch.int64,
                device=index_scores.device,
            )
            pool_topk = torch.cat([pool_topk, pad], dim=-1)
    else:
        if mask is not None:
            assert mask.dtype == torch.float32, "mask dtype must be float32"
        index_scores = None
        pool_topk = None
        if pool_topk_fn is not None and mask is None and not use_per_seg:
            pool_topk = pool_topk_fn(
                q, k_pooled, weights, index_topk // pool_size, pool_size, hf_scoring
            )
        if pool_topk is None:
            pool_topk = _kpool_topk_from_score_chunks(
                q=q,
                k_pooled=k_pooled,
                weights=weights,
                index_topk=index_topk,
                pool_size=pool_size,
                pool_exists=pool_exists if use_per_seg else None,
                mask=mask,
                varlen_starts=v_starts,
                varlen_ends=v_ends,
                key_positions=k_pos,
                pool_token_base=pool_token_base,
                sk=sk,
                use_relu=use_relu,
                hf_scoring=hf_scoring,
            )

    # Expand [batch * queries, pools] to a fixed token budget.
    rows = pool_topk.shape[0] * pool_topk.shape[1]
    pool_flat = pool_topk.reshape(rows, -1)
    pool_valid = pool_flat >= 0
    # Clamp invalid ids to 0 for the arithmetic, restore -1 via the where mask.
    safe_pool = pool_flat.clamp(min=0)
    token_topk = _expand_pools_to_tokens(
        safe_pool,
        pool_valid,
        index_topk,
        pool_size,
        pool_token_base=pool_token_base if use_per_seg else None,
    )
    # token_topk is [rows, index_topk]; reshape back to [batch, seqlen_q, index_topk].
    token_topk = token_topk.reshape(pool_topk.shape[0], pool_topk.shape[1], index_topk)

    if always_select_tail:
        # Pool phase is query-local, never the final length of the packed sample.
        sq, batch = q.shape[:2]
        if v_ends is not None:
            ends = v_ends
        elif mask is not None:
            mask_starts, ends = _query_bounds_from_token_mask(mask)
        else:
            ends = torch.arange(1, sq + 1, device=q.device)
        if v_starts is not None:
            starts = v_starts
        elif mask is not None:
            starts = mask_starts
        elif cu_seqlens_kv is not None:
            cu = cu_seqlens_kv.to(device=q.device, dtype=torch.int64)
            starts = cu[torch.searchsorted(cu[1:], ends - 1, right=True)]
        else:
            starts = torch.zeros_like(ends)
        lengths = ends - starts
        if mask is not None and v_starts is None:
            # Pools are aligned to the shifted key axis. Fill a leading partial pool, which
            # only a row starting off the grid has, into unused expanded-pool slots, then
            # append the trailing tail on the same grid.
            starts = _expand_query_bounds_for_batch(starts, batch, sq).to(
                device=token_topk.device, dtype=torch.int64
            )
            ends = _expand_query_bounds_for_batch(ends, batch, sq).to(
                device=token_topk.device, dtype=torch.int64
            )
            token_topk = _fill_kpool_left_boundary(
                token_topk.reshape(rows, -1), starts.reshape(-1), ends.reshape(-1), pool_size
            ).reshape(batch, sq, -1)
            complete_end = (ends // pool_size) * pool_size
            tail_starts = complete_end
            tail_counts = (ends - tail_starts).clamp_min(0).clamp_max(pool_size - 1)
        else:
            starts = _expand_query_bounds_for_batch(starts, batch, sq)
            ends = _expand_query_bounds_for_batch(ends, batch, sq)
            lengths = ends - starts
            tail_starts = ends - lengths.remainder(pool_size)
            tail_counts = None
        token_topk = _append_tail_to_topk(
            token_topk.reshape(rows, -1),
            lengths.reshape(-1),
            pool_size,
            tail_start_override=tail_starts.reshape(-1),
            tail_count_override=tail_counts.reshape(-1) if tail_counts is not None else None,
        ).reshape(batch, sq, -1)

    if pool_phase is not None:
        phase = pool_phase.view(-1, 1, 1).to(token_topk.dtype)
        token_topk = torch.where(token_topk >= 0, token_topk + phase, token_topk)
    if token_mask is not None:
        token_topk = _mask_topk_tokens_with_token_mask(token_topk, token_mask)

    if return_selection:
        return index_scores, token_topk, pool_topk.to(torch.int64)
    return index_scores, token_topk


def _merge_topk_scores(
    running_scores: Optional[torch.Tensor],
    running_indices: Optional[torch.Tensor],
    block_scores: torch.Tensor,
    block_indices: torch.Tensor,
    topk_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Merge two candidate top-k sets into an exact top-k set."""
    if running_scores is None or running_indices is None:
        return block_scores, block_indices

    merged_scores = torch.cat((running_scores, block_scores), dim=-1)
    merged_indices = torch.cat((running_indices, block_indices), dim=-1)
    keep_k = min(topk_k, merged_scores.size(-1))
    keep = merged_scores.topk(keep_k, dim=-1)[1]
    running_scores = torch.gather(merged_scores, -1, keep)
    running_indices = torch.gather(merged_indices, -1, keep)
    return running_scores, running_indices


def fused_qk_topk_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    index_topk: int,
    mask: Optional[torch.Tensor] = None,
    key_chunk_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact top-k routing over key chunks.

    Returns the exact same top-k result as the dense implementation, but avoids materializing the
    full score tensor when `key_chunk_size` is set.
    """
    sk = k.size(0)
    topk_k = min(index_topk, sk)
    if key_chunk_size is None or key_chunk_size <= 0 or key_chunk_size >= sk:
        index_scores, topk_indices = fused_qk_topk_naive(q, k, weights, index_topk, mask)
        topk_scores = torch.gather(index_scores, -1, topk_indices)
        return topk_scores, topk_indices

    running_scores = None
    running_indices = None
    for k_start in range(0, sk, key_chunk_size):
        k_end = min(k_start + key_chunk_size, sk)
        block_scores = _compute_index_scores(q, weights, k[k_start:k_end])
        if mask is not None:
            block_mask = mask[..., k_start:k_end]
            assert (
                block_mask.dtype == block_scores.dtype
            ), "Mask dtype must match index scores dtype"
            block_scores = block_scores + block_mask

        block_topk_k = min(topk_k, k_end - k_start)
        block_scores, block_indices = block_scores.topk(block_topk_k, dim=-1)
        block_indices = block_indices + k_start
        running_scores, running_indices = _merge_topk_scores(
            running_scores, running_indices, block_scores, block_indices, topk_k
        )

    return running_scores, running_indices


def fwd_fused_indexer_loss_naive(
    q,
    weights,
    k,
    query,
    key,
    topk,
    softmax_scale,
    loss_coeff,
    mask,
    sparse_loss,
    pg_collection,
    varlen_starts=None,
    varlen_ends=None,
    key_positions=None,
    query_valid_rows=None,
    calculate_per_token_loss: bool = False,
    use_relu: bool = True,
    non_compressed_lse: torch.Tensor | None = None,
):
    """Naive implementation of forward pass for indexer loss."""
    index_scores, topk_indices = fused_qk_topk_naive(
        q,
        k,
        weights,
        topk,
        mask=mask,
        varlen_starts=varlen_starts,
        varlen_ends=varlen_ends,
        key_positions=key_positions,
        use_relu=use_relu,
    )

    indexer_loss = compute_dsa_indexer_loss(
        index_scores,
        topk_indices,
        query,
        key,
        softmax_scale,
        loss_coeff,
        sparse_loss,
        pg_collection,
        mask=mask,
        varlen_starts=varlen_starts,
        varlen_ends=varlen_ends,
        key_positions=key_positions,
        query_valid_rows=query_valid_rows,
        calculate_per_token_loss=calculate_per_token_loss,
        non_compressed_lse=non_compressed_lse,
    )

    return topk_indices, indexer_loss


def bwd_fused_indexer_loss_naive(
    q,
    weights,
    k,
    query,
    key,
    topk_indices,
    softmax_scale,
    loss_coeff,
    sparse_loss,
    mask,
    grad_loss,
    pg_collection,
    varlen_starts=None,
    varlen_ends=None,
    key_positions=None,
    query_valid_rows=None,
    calculate_per_token_loss: bool = False,
    use_relu: bool = True,
    non_compressed_lse: torch.Tensor | None = None,
):
    """Naive implementation of backward pass for indexer loss."""
    query, _ = dsa_layout.ensure_sbhd(query, "query")
    key, _ = dsa_layout.ensure_sbhd(key, "key")

    index_scores = _compute_index_scores(q, weights, k, use_relu=use_relu)  # [B, Sq, Sk]

    sq, b, np, hn = query.size()
    sk = key.size(0)
    query_valid_rows = dsa_masking.normalize_query_valid_rows(
        query_valid_rows, b=b, sq=sq, device=query.device
    )

    # [sq, b, np, hn] -> [b, np, sq, hn] -> [b * np, sq, hn]
    query_reshaped = query.permute(1, 2, 0, 3).reshape(b * np, sq, hn)
    # [sk, b, np, hn] -> [b, np, hn, sk] -> [b * np, hn, sk]
    key_reshaped = key.permute(1, 2, 3, 0).reshape(b * np, hn, sk)
    # Compute attention scores [b * np, sq, sk]
    attention_scores = torch.bmm(query_reshaped.float(), key_reshaped.float()) * softmax_scale
    # Free reshaped tensors - no longer needed after bmm
    del query_reshaped, key_reshaped

    # Reshape to [b, np, sq, sk]
    attention_scores = attention_scores.reshape(b, np, sq, sk)
    varlen_starts, varlen_ends, key_positions = dsa_masking.normalize_varlen_bounds(
        mask=mask,
        varlen_starts=varlen_starts,
        varlen_ends=varlen_ends,
        key_positions=key_positions,
        sk=sk,
        device=attention_scores.device,
    )

    if varlen_starts is not None:
        attention_scores = dsa_masking.apply_starts_ends_mask_to_scores(
            attention_scores, varlen_starts, varlen_ends, key_positions
        )
        index_scores = dsa_masking.apply_starts_ends_mask_to_scores(
            index_scores, varlen_starts, varlen_ends, key_positions
        )
        base_valid_mask = (
            dsa_masking.build_valid_mask_from_starts_ends(varlen_starts, varlen_ends, key_positions)
            .unsqueeze(0)
            .expand(b, sq, sk)
        )
    else:
        _, attn_score_mask, index_score_mask, base_valid_mask = dsa_masking.prepare_additive_mask(
            mask, sq=sq, sk=sk, b=b, device=attention_scores.device
        )
        # [b, np, sq, sk] + [1/b, 1, sq, sk] -> [b, np, sq, sk]
        attention_scores = attention_scores + attn_score_mask
        # [b, sq, sk] + [1/b, sq, sk] -> [b, sq, sk]
        index_scores = index_scores + index_score_mask

    # index_mask [b, sq, sk]
    index_mask = torch.full(
        (b, sq, sk), float("-inf"), dtype=torch.float32, device=attention_scores.device
    )
    dsa_masking.scatter_topk_into_index_mask(index_mask, topk_indices, seq_chunk_size=256)

    if sparse_loss:
        # [b, np, sq, sk] + [b, 1, sq, sk] -> [b, np, sq, sk]
        attention_scores = attention_scores + index_mask.view(b, 1, sq, sk)
        # [b, sq, sk] + [b, sq, sk] -> [b, sq, sk]
        index_scores = index_scores + index_mask

    # Compute softmax for both.
    if sparse_loss:
        index_valid_mask = base_valid_mask & (index_mask == 0)
    else:
        index_valid_mask = base_valid_mask
    attention_valid_mask = index_valid_mask if sparse_loss else base_valid_mask
    attention_scores_softmax = _compute_indexer_teacher_probabilities(
        attention_scores, attention_valid_mask, non_compressed_lse=non_compressed_lse
    )
    # Free attention_scores immediately
    del attention_scores

    index_scores_softmax = dsa_masking.masked_softmax(
        index_scores.float(), index_valid_mask, dim=-1
    )
    # Free index_scores - no longer needed after softmax
    del index_scores

    # Sum attention scores across heads: [b, np, sq, sk] -> [b, sq, sk]
    attention_scores_sum = attention_scores_softmax.sum(dim=1)
    # Free attention_scores_softmax
    del attention_scores_softmax

    if pg_collection.tp.size() > 1:
        # attention scores are scattered to TP ranks in head dimension.
        torch.distributed.all_reduce(attention_scores_sum.contiguous(), group=pg_collection.tp)

    # L1 normalize. Fully masked packed/varlen rows can have zero summed
    # attention mass; clamp the denominator so those rows stay finite and are
    # later zeroed by the row-valid loss mask.
    attention_scores_normalized = _normalize_indexer_teacher_target(
        attention_scores_sum, non_compressed_lse
    )
    # Free attention_scores_sum - no longer needed after normalization
    del attention_scores_sum

    # Backward through loss = kl_div * loss_coeff
    # where kl_div = kl_per_element.sum(dim=-1).mean()
    grad_kl_div = grad_loss * loss_coeff  # scalar

    if calculate_per_token_loss:
        grad_kl_per_row = grad_kl_div
    else:
        valid_row_count = (
            query_valid_rows.sum().to(
                dtype=torch.float32, device=attention_scores_normalized.device
            )
            if query_valid_rows is not None
            else torch.tensor(
                float(b * sq), dtype=torch.float32, device=attention_scores_normalized.device
            )
        ).clamp_min(1.0)
        grad_kl_per_row = grad_kl_div / valid_row_count  # scalar value for each real row

    # Backward through sum(dim=-1): broadcast back to [b, sq, sk]
    # Each element in a row contributes to the sum, so gradient is same for all
    grad_kl_per_element = grad_kl_per_row.view(1, 1, 1).expand(b, sq, sk)
    if query_valid_rows is not None:
        grad_kl_per_element = grad_kl_per_element * query_valid_rows.unsqueeze(-1).to(
            dtype=grad_kl_per_element.dtype
        )

    # For KL(target || softmax(logits)), the exact logit gradient is
    # predict * target.sum(-1) - target. Positive teacher rows are L1-normalized,
    # while a fully masked zero-mass row must have zero gradient.
    attention_target_mass = attention_scores_normalized.sum(dim=-1, keepdim=True)
    grad_index_scores_logits = (
        index_scores_softmax * attention_target_mass - attention_scores_normalized
    ) * grad_kl_per_element
    del index_scores_softmax, attention_scores_normalized

    # Zero out gradients for masked positions.
    if sparse_loss:
        # Also apply index mask - only topk positions are valid.
        del index_mask
        valid_mask = base_valid_mask & index_valid_mask  # [b, sq, sk]
        del index_valid_mask
    else:
        del index_mask
        valid_mask = base_valid_mask  # [b, sq, sk]
    del base_valid_mask
    if query_valid_rows is not None:
        valid_mask = valid_mask & query_valid_rows.unsqueeze(-1)

    grad_index_scores_logits = grad_index_scores_logits * valid_mask.float()
    del valid_mask

    # Transpose from [b, sq, sk] to [sq, b, sk]
    grad_index_scores = grad_index_scores_logits.transpose(0, 1)  # [sq, b, sk]
    del grad_index_scores_logits

    # Backward through sum over heads: expand gradient
    grad_weighted_scores = grad_index_scores.unsqueeze(2)  # [sq, b, 1, sk]
    del grad_index_scores

    # Chunk over seqlen_q to avoid materializing the full [sq, b, h, sk] fp32 tensor.
    sq_q, b_q, h_q, d_q = q.shape
    if sq_q == 0:
        return (torch.zeros_like(q), torch.zeros_like(weights), torch.zeros_like(k))
    k_fp32 = k.float()
    bytes_per_token = b_q * h_q * sk * 4
    chunk_size = min(sq_q, max(1, 1024 * 1024 * 1024 // max(1, bytes_per_token)))

    grad_q = torch.empty(sq_q, b_q, h_q, d_q, dtype=torch.float32, device=q.device)
    grad_weights = torch.empty(sq_q, b_q, h_q, dtype=torch.float32, device=q.device)
    grad_k = torch.zeros(sk, b_q, d_q, dtype=torch.float32, device=q.device)

    for start in range(0, sq_q, chunk_size):
        end = min(start + chunk_size, sq_q)
        q_chunk = q[start:end]  # [chunk, b, h, d]
        gw_chunk = grad_weighted_scores[start:end]  # [chunk, b, 1, sk]
        w_chunk = weights[start:end]  # [chunk, b, h]

        # Forward scores for this chunk: [chunk, b, h, sk]
        scores_chunk = torch.einsum('sbhd,tbd->sbht', q_chunk.float(), k_fp32)
        if use_relu:
            relu_mask_chunk = scores_chunk > 0
            scores_chunk.relu_()
        else:
            relu_mask_chunk = None

        # ∂L/∂weights = grad * scores (sum over sk)
        grad_weights[start:end] = (gw_chunk * scores_chunk).sum(dim=-1)
        del scores_chunk

        # ∂L/∂scores = grad * weights: [chunk, b, h, sk]
        grad_scores_chunk = gw_chunk * w_chunk.unsqueeze(-1)
        if use_relu:
            grad_scores_chunk.masked_fill_(~relu_mask_chunk, 0.0)
            del relu_mask_chunk

        # ∂L/∂q = einsum('sbht,tbd->sbhd', grad_scores, k)
        grad_q[start:end] = torch.einsum('sbht,tbd->sbhd', grad_scores_chunk, k_fp32)
        # ∂L/∂k = einsum('sbht,sbhd->tbd', grad_scores, q)  (accumulate)
        grad_k += torch.einsum('sbht,sbhd->tbd', grad_scores_chunk, q_chunk.float())
        del grad_scores_chunk

    del grad_weighted_scores

    return grad_q.to(q.dtype), grad_weights.to(weights.dtype), grad_k.to(k.dtype)


_FUSED_DSA_INDEXER_LOSS_INPUT_NAMES = (
    "q",
    "weights",
    "k",
    "query",
    "key",
    "softmax_scale",
    "topk",
    "loss_coeff",
    "mask",
    "sparse_loss",
    "pg_collection",
    "varlen_starts",
    "varlen_ends",
    "key_positions",
    "query_valid_rows",
    "calculate_per_token_loss",
    "use_relu",
    "non_compressed_lse",
)


class FusedDSAIndexerLoss(torch.autograd.Function):
    """Fused implementation of DSA Indexer Loss."""

    @staticmethod
    def forward(
        ctx,
        q,
        weights,
        k,
        query,
        key,
        softmax_scale,
        topk,
        loss_coeff,
        mask,
        sparse_loss,
        pg_collection,
        varlen_starts=None,
        varlen_ends=None,
        key_positions=None,
        query_valid_rows=None,
        calculate_per_token_loss: bool = False,
        use_relu: bool = True,
        non_compressed_lse: torch.Tensor | None = None,
    ):
        """
        Fused forward: index_scores never materialized in full.
        """
        topk_indices, loss = fwd_fused_indexer_loss_naive(
            q,
            weights,
            k,
            query,
            key,
            topk,
            softmax_scale,
            loss_coeff,
            mask,
            sparse_loss,
            pg_collection,
            varlen_starts=varlen_starts,
            varlen_ends=varlen_ends,
            key_positions=key_positions,
            query_valid_rows=query_valid_rows,
            calculate_per_token_loss=calculate_per_token_loss,
            use_relu=use_relu,
            non_compressed_lse=non_compressed_lse,
        )

        # Save for backward (recomputation strategy)
        saved_non_compressed_lse = (
            non_compressed_lse
            if non_compressed_lse is not None
            else q.new_empty(0, dtype=torch.float32)
        )
        ctx.save_for_backward(q, weights, k, query, key, topk_indices, saved_non_compressed_lse)
        ctx.has_non_compressed_lse = non_compressed_lse is not None
        ctx.softmax_scale = softmax_scale
        ctx.loss_coeff = loss_coeff
        ctx.sparse_loss = sparse_loss
        ctx.mask = mask
        ctx.pg_collection = pg_collection
        ctx.varlen_starts = varlen_starts
        ctx.varlen_ends = varlen_ends
        ctx.key_positions = key_positions
        ctx.query_valid_rows = query_valid_rows
        ctx.calculate_per_token_loss = calculate_per_token_loss
        ctx.use_relu = use_relu
        ctx.num_inputs = len(ctx.needs_input_grad)

        return topk_indices, loss

    @staticmethod
    def backward(ctx, grad_topk_indices, grad_loss):
        """
        Backward: Recompute what we need.
        """
        q, weights, k, query, key, topk_indices, saved_non_compressed_lse = ctx.saved_tensors
        non_compressed_lse = saved_non_compressed_lse if ctx.has_non_compressed_lse else None

        grad_q, grad_weights, grad_k = bwd_fused_indexer_loss_naive(
            q,
            weights,
            k,
            query,
            key,
            topk_indices,
            ctx.softmax_scale,
            ctx.loss_coeff,
            ctx.sparse_loss,
            ctx.mask,
            grad_loss,
            ctx.pg_collection,
            varlen_starts=ctx.varlen_starts,
            varlen_ends=ctx.varlen_ends,
            key_positions=ctx.key_positions,
            query_valid_rows=ctx.query_valid_rows,
            calculate_per_token_loss=ctx.calculate_per_token_loss,
            use_relu=ctx.use_relu,
            non_compressed_lse=non_compressed_lse,
        )

        grad_by_name = {
            "q": grad_q,
            "weights": grad_weights,
            "k": grad_k,
            # query and key are detached in forward, so return None for their gradients.
            "query": None,
            "key": None,
            "non_compressed_lse": None,
        }
        gradients = tuple(grad_by_name.get(name) for name in _FUSED_DSA_INDEXER_LOSS_INPUT_NAMES)
        return gradients[: ctx.num_inputs]


# Byte budget of one row chunk of the KPool sparse indexer loss (teacher and student).
_KPOOL_LOSS_CHUNK_BYTES = 1024 * 1024 * 1024


def _kpool_loss_chunk_rows(rows: int, bytes_per_row: int, chunk_bytes: int) -> int:
    """Query rows per chunk, from shapes only (no host sync)."""
    return min(rows, max(1, int(chunk_bytes) // max(1, int(bytes_per_row))))


def _kpool_canonical_slots(
    token_topk: torch.Tensor, pool_ids: torch.Tensor, *, index_topk: int, pool_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Put a KPool selection in canonical slot order: valid pools by ascending pool id.

    ``pool_ids`` is ``[batch, sq, index_topk // pool_size]`` (``-1`` for unused slots) in
    the selector's slot order, and ``token_topk`` is ``[batch, sq, index_topk + tail]``,
    where pool slot ``j`` holds token slots ``j * pool_size .. j * pool_size + pool_size - 1``
    and the last ``tail`` columns are the query-local tail. A stable sort of the valid pool
    ids (unused slots last) regroups both the same way and keeps the tail columns in place,
    so ranks that selected the same pools hold them in the same slots. The inputs are not
    modified; new tensors are returned.

    Returns:
        ``(pool_ids_c, token_topk_c)`` with the shapes and dtypes of the inputs.
    """
    budget = index_topk // pool_size
    if pool_ids.ndim != 3 or pool_ids.size(-1) != budget:
        raise ValueError(f"pool_ids must be [batch, sq, {budget}], got {tuple(pool_ids.shape)}.")
    if token_topk.ndim != 3 or token_topk.shape[:2] != pool_ids.shape[:2]:
        raise ValueError(
            "token_topk must be [batch, sq, width] with the rows of pool_ids, got "
            f"{tuple(token_topk.shape)} and {tuple(pool_ids.shape)}."
        )
    if token_topk.size(-1) < index_topk:
        raise ValueError(
            f"token_topk must have at least index_topk={index_topk} columns, got "
            f"{token_topk.size(-1)}."
        )
    batch, sq = pool_ids.shape[:2]
    unused_last = torch.full_like(pool_ids, torch.iinfo(pool_ids.dtype).max)
    order = torch.sort(
        torch.where(pool_ids >= 0, pool_ids, unused_last), dim=-1, stable=True
    ).indices
    pool_ids_c = torch.gather(pool_ids, -1, order)
    pool_tokens = token_topk[..., :index_topk].reshape(batch, sq, budget, pool_size)
    pool_tokens = torch.gather(pool_tokens, 2, order.unsqueeze(-1).expand(-1, -1, -1, pool_size))
    token_topk_c = torch.cat(
        (pool_tokens.reshape(batch, sq, index_topk), token_topk[..., index_topk:]), dim=-1
    )
    return pool_ids_c, token_topk_c


@torch.no_grad()
def _kpool_sparse_pool_target(
    query: torch.Tensor,
    key: torch.Tensor,
    token_topk_c: torch.Tensor,
    pool_ids_c: torch.Tensor,
    *,
    softmax_scale: float,
    index_topk: int,
    pool_size: int,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
    chunk_bytes: int = _KPOOL_LOSS_CHUNK_BYTES,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pool-level teacher of the sparse KPool indexer loss, from the main attention.

    For each main-attention head ``h`` on this rank and query row ``t``, with the attended
    token ids ``A(t)`` of ``token_topk_c`` (selected pools plus the query-local tail;
    ``-1`` is ignored):

    - ``lse_h(t) = logsumexp_{j in A(t)} softmax_scale * query_h(t) . key_j``, the
      attention's own normalizer, tail included;
    - ``m_h(t, c)`` is the sum of ``exp(logit - lse_h(t))`` over the ``pool_size`` tokens of
      the selected pool in slot ``c``: an absolute mass with no common shift.

    The masses are summed over heads, SUM all-reduced over ``tp_group`` (which shards the
    heads; every rank must hold the same canonical slots) and L1-normalized over the slots
    with an exact-zero safe divide. Rows whose summed pool mass is exactly zero (no complete
    pool, or a pool mass that underflows FP32 on every head) get an all-zero target. Unused
    slots (``pool_ids_c < 0``) get zero target. The math is FP32 in row chunks of about
    ``chunk_bytes``, sized from shapes without a host sync. On CUDA, BF16 or FP16 query and
    latent key go through one GEMM with FP32 accumulation and FP32 output, which skips an
    FP32 copy of the gathered keys (the products of two BF16 or FP16 values are exact in
    FP32); other inputs are upcast to FP32 first.

    Args:
        query: Detached main-attention query ``[sq, batch, heads, dim]``.
        key: Detached main-attention key ``[sk, batch, 1 or heads, dim]`` (one head for the
            absorbed-MLA latent key).
        token_topk_c: Attended token ids ``[batch, sq, index_topk + tail]`` in canonical
            slot order (see ``_kpool_canonical_slots``).
        pool_ids_c: Selected pool ids ``[batch, sq, index_topk // pool_size]``, canonical.

    Returns:
        ``(target [batch, sq, pools] fp32, row_has_mass [batch, sq] bool,
        tail_mass [batch, sq] fp32)``. ``tail_mass`` is the head-summed probability of the
        tail divided by the head-summed probability of every attended token (``p_E``),
        zero for rows without attended tokens.
    """
    query, _ = dsa_layout.ensure_sbhd(query, "query")
    key, _ = dsa_layout.ensure_sbhd(key, "key")
    sq, batch, n_heads, head_dim = query.shape
    sk, key_batch, key_heads, key_dim = key.shape
    if key_batch != batch or key_dim != head_dim or key_heads not in (1, n_heads):
        raise ValueError(
            "KPool teacher expects key [sk, batch, 1 or heads, dim] matching query "
            f"{tuple(query.shape)}, got {tuple(key.shape)}."
        )
    budget = index_topk // pool_size
    if pool_ids_c.shape != (batch, sq, budget):
        raise ValueError(
            f"pool_ids_c must be [{batch}, {sq}, {budget}], got {tuple(pool_ids_c.shape)}."
        )
    width = token_topk_c.size(-1)
    if token_topk_c.shape[:2] != (batch, sq) or width < index_topk:
        raise ValueError(
            f"token_topk_c must be [{batch}, {sq}, >= {index_topk}], got "
            f"{tuple(token_topk_c.shape)}."
        )
    device = query.device
    # Per row: head-summed pool masses, then the tail mass and the total attended mass.
    # One contiguous buffer, so a single all-reduce sums the TP-sharded heads in place.
    masses = torch.zeros(batch, sq, budget + 2, dtype=torch.float32, device=device)
    if sq > 0 and sk > 0:
        key_flat = key.reshape(sk * batch, key_heads * key_dim)
        low_precision_gemm = (
            device.type == "cuda"
            and key_heads == 1
            and query.dtype == key.dtype
            and query.dtype in (torch.bfloat16, torch.float16)
        )
        gathered_key_bytes = key.element_size() + (0 if low_precision_gemm else 4)
        bytes_per_row = batch * (
            width * key_heads * key_dim * gathered_key_bytes
            + 3 * n_heads * width * 4
            + n_heads * head_dim * 4
            + width * 8 * 3
        )
        chunk = _kpool_loss_chunk_rows(sq, bytes_per_row, chunk_bytes)
        batch_offsets = torch.arange(batch, device=device, dtype=torch.int64).view(batch, 1, 1)
        for start in range(0, sq, chunk):
            end = min(start + chunk, sq)
            rows = end - start
            ids = token_topk_c[:, start:end].to(torch.int64)
            valid = (ids >= 0) & (ids < sk)
            flat_ids = ids.clamp(0, sk - 1) * batch + batch_offsets
            # [batch, rows, width, key_heads, dim]
            keys = key_flat.index_select(0, flat_ids.reshape(-1)).view(
                batch, rows, width, key_heads, key_dim
            )
            queries = query[start:end].permute(1, 0, 2, 3)  # [batch, rows, heads, dim]
            with torch.autocast(device_type=device.type, enabled=False):
                if low_precision_gemm:
                    logits = torch.bmm(
                        queries.reshape(batch * rows, n_heads, head_dim),
                        keys.view(batch * rows, width, key_dim).transpose(1, 2),
                        out_dtype=torch.float32,
                    ).view(batch, rows, n_heads, width)
                elif key_heads == 1:
                    logits = torch.matmul(
                        queries.float(), keys.squeeze(3).float().transpose(-1, -2)
                    )
                else:
                    logits = torch.einsum("brhd,brwhd->brhw", queries.float(), keys.float())
            del keys
            logits.mul_(softmax_scale).masked_fill_(~valid.unsqueeze(2), float("-inf"))
            lse = torch.logsumexp(logits, dim=-1, keepdim=True)
            # Rows without an attended token keep -inf logits, so their mass is exactly zero.
            lse = torch.where(torch.isfinite(lse), lse, torch.zeros_like(lse))
            probs = logits.sub_(lse).exp_()  # [batch, rows, heads, width]
            del logits, lse
            # Every mass below is a sum over heads: take it first, on [batch, rows, width].
            head_mass = probs.sum(dim=2)
            del probs
            pool_mass = head_mass[..., :index_topk].reshape(batch, rows, budget, pool_size).sum(-1)
            pool_mass.masked_fill_(pool_ids_c[:, start:end] < 0, 0.0)
            masses[:, start:end, :budget] = pool_mass
            masses[:, start:end, budget] = head_mass[..., index_topk:].sum(-1)
            masses[:, start:end, budget + 1] = head_mass.sum(-1)
            del head_mass, pool_mass
    if tp_group is not None and get_pg_size(tp_group) > 1:
        torch.distributed.all_reduce(masses, group=tp_group)
    pool_mass = masses[..., :budget]
    target = _normalize_teacher_target_exact_zero(pool_mass)
    row_has_mass = pool_mass.sum(dim=-1) > 0
    total_mass = masses[..., budget + 1]
    tail_mass = masses[..., budget] / torch.where(
        total_mass > 0, total_mass, torch.ones_like(total_mass)
    )
    return target, row_has_mass, tail_mass


def _kpool_gather_selected_pool_keys(
    k_pooled: torch.Tensor, pool_ids: torch.Tensor
) -> torch.Tensor:
    """Gather ``k_pooled [pools, batch, dim]`` at ``pool_ids [batch, rows, slots]``.

    Returns ``[batch, rows, slots, dim]`` in the key dtype; unused slots (``-1``) read
    pool 0, and zeros are returned when there are no pools.
    """
    num_pools, batch, head_dim = k_pooled.shape
    if num_pools == 0:
        return k_pooled.new_zeros(pool_ids.shape + (head_dim,))
    flat_ids = pool_ids.clamp(0, num_pools - 1).to(torch.int64) * batch + torch.arange(
        batch, device=pool_ids.device, dtype=torch.int64
    ).view(batch, 1, 1)
    return (
        k_pooled.reshape(num_pools * batch, head_dim)
        .index_select(0, flat_ids.reshape(-1))
        .view(pool_ids.shape + (head_dim,))
    )


def _kpool_student_dtype(*tensors: torch.Tensor) -> torch.dtype:
    """Arithmetic dtype of the KPool student: FP32, or FP64 for FP64 inputs (gradcheck)."""
    return torch.float64 if any(t.dtype == torch.float64 for t in tensors) else torch.float32


def _kpool_selected_pool_head_scores(
    q: torch.Tensor,
    k_pooled: torch.Tensor,
    pool_ids: torch.Tensor,
    *,
    hf_scoring: bool,
    use_relu: bool = True,
    compute_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-head selected-pool scores, before the head-weight reduction.

    Returns ``(head_scores [batch, rows, heads, slots], queries [batch, rows, heads, dim],
    keys [batch, rows, slots, dim])`` in ``compute_dtype`` (FP32 unless an input is FP64).
    With ``hf_scoring`` the scores carry the QK scale ``dim**-0.5`` inside the ReLU, as
    ``_compute_kpool_hf_scores`` does.
    """
    if hf_scoring and not use_relu:
        raise ValueError("HF KPool scoring requires use_relu=True.")
    if compute_dtype is None:
        compute_dtype = _kpool_student_dtype(q, k_pooled)
    with torch.autocast(device_type=q.device.type, enabled=False):
        keys = _kpool_gather_selected_pool_keys(k_pooled, pool_ids).to(compute_dtype)
        queries = q.permute(1, 0, 2, 3).to(compute_dtype)
        scores = torch.matmul(queries, keys.transpose(-1, -2))
        if hf_scoring:
            scores = torch.relu(scores * (q.size(-1) ** -0.5))
        elif use_relu:
            scores = torch.relu(scores)
    return scores, queries, keys


def _kpool_reduce_selected_pool_head_scores(
    head_scores: torch.Tensor, weights: torch.Tensor, *, hf_scoring: bool
) -> torch.Tensor:
    """Sum per-head scores with the head weights ``[rows, batch, heads]``, in selector order.

    HF scoring uses the head-reduction GEMM of ``_compute_kpool_hf_scores``; legacy scoring
    multiplies and sums over heads, as ``_kpool_topk_from_score_chunks`` does. The weights
    take the dtype of ``head_scores``.
    """
    with torch.autocast(device_type=head_scores.device.type, enabled=False):
        head_weights = weights.permute(1, 0, 2).to(head_scores.dtype)  # [batch, rows, heads]
        if hf_scoring:
            return torch.matmul(head_weights.unsqueeze(-2), head_scores).squeeze(-2)
        return (head_scores * head_weights.unsqueeze(-1)).sum(dim=2)


def _kpool_selected_pool_scores(
    q: torch.Tensor,
    weights: torch.Tensor,
    k_pooled: torch.Tensor,
    pool_ids: torch.Tensor,
    *,
    hf_scoring: bool,
    use_relu: bool = True,
) -> torch.Tensor:
    """Student score of the sparse KPool indexer loss: the selector's score of selected pools.

    ``q`` is ``[rows, batch, heads, dim]``, ``weights`` ``[rows, batch, heads]``,
    ``k_pooled`` ``[pools, batch, dim]`` (the scoring inputs after any rotation) and
    ``pool_ids`` ``[batch, rows, slots]``. With ``hf_scoring``,
    ``I = sum_h w_h * relu((q_h . k) * dim**-0.5)``, where ``weights`` carries only the
    head-count scale; otherwise ``I = sum_h w_h * relu(q_h . k)`` (ReLU per ``use_relu``),
    with every scale folded into ``weights``. This is the single definition of the student
    score. Unused slots (``-1``) get an arbitrary finite value; callers mask them.

    Returns:
        ``I [batch, rows, slots]`` in FP32 (FP64 when an input is FP64).
    """
    head_scores, _, _ = _kpool_selected_pool_head_scores(
        q,
        k_pooled,
        pool_ids,
        hf_scoring=hf_scoring,
        use_relu=use_relu,
        compute_dtype=_kpool_student_dtype(q, weights, k_pooled),
    )
    return _kpool_reduce_selected_pool_head_scores(head_scores, weights, hf_scoring=hf_scoring)


def _kpool_student_bytes_per_row(batch: int, n_heads: int, head_dim: int, budget: int) -> int:
    """Upper estimate of the student forward/backward bytes per query row (all samples)."""
    return batch * (
        budget * head_dim * 32  # gathered keys (native + FP32), dK slots, sort copy, FP64 prefix
        + n_heads * budget * 4 * 4  # head scores, ReLU mask, head-score gradients
        + n_heads * head_dim * 4 * 2  # FP32 queries and their gradient
        + budget * 8 * 4  # slot ids, sort keys and orders
    )


def _kpool_accumulate_pool_key_grads(
    grad_k_ext: torch.Tensor,
    flat_ids: torch.Tensor,
    contributions: torch.Tensor,
    key_ids: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Add per-slot pooled-key gradients into ``grad_k_ext`` without floating-point atomics.

    With ``torch.are_deterministic_algorithms_enabled()``, use torch's deterministic
    ``index_add_`` (sort based, fixed order). Otherwise sort the slots by key id (stable)
    and sum each key's run of slots with ``torch.segment_reduce``, which adds a segment's
    elements in a fixed order; the segment lengths come from ``searchsorted`` on the sorted
    ids, so nothing waits on the host. A run holds at most one slot per row of the chunk
    (a row's pools are distinct, and the caller spreads unused slots over padding rows), so
    each segment stays short. Both orders are fixed, as in CSA's
    ``_deterministic_sparse_indexer_grads_wk``. (A prefix sum over the sorted slots does the
    same with more passes, and PyTorch's CUDA scan over a leading dimension runs one serial
    thread per column: 0.1 s per chunk at 1M tokens.) Returns the (lazily built)
    ``key_ids`` for reuse by the next chunk.
    """
    if torch.are_deterministic_algorithms_enabled():
        grad_k_ext.index_add_(0, flat_ids, contributions)
        return key_ids
    if key_ids is None:
        key_ids = torch.arange(grad_k_ext.size(0), device=grad_k_ext.device, dtype=torch.int64)
    sorted_ids, order = torch.sort(flat_ids, stable=True)
    lengths = torch.searchsorted(sorted_ids, key_ids, right=True)
    lengths.sub_(torch.searchsorted(sorted_ids, key_ids, right=False))
    # unsafe=True skips the length checks, which read the lengths on the host; the lengths
    # sum to the slot count by construction (every id is a row of grad_k_ext).
    grad_k_ext.add_(
        torch.segment_reduce(
            contributions.index_select(0, order), "sum", lengths=lengths, axis=0, unsafe=True
        )
    )
    return key_ids


class KPoolSparseIndexerLoss(torch.autograd.Function):
    """Sparse pool-level KL loss that trains a KPool DSA indexer.

    ``loss = loss_coeff * sum_rows(row_mask * KL(target || masked_softmax(I))) / row_count``
    with ``I = _kpool_selected_pool_scores(q, weights, k_pooled, pool_ids_c)`` over the
    selected pools (unused ``-1`` slots excluded) and the token loss's reduction: the
    caller passes the token loss's row count, or 1 under ``calculate_per_token_loss``.
    The forward saves only its inputs and recomputes the scores per row chunk in the
    backward, which applies the exact logit gradient ``(p * sum(t) - t) * scale`` and
    backpropagates it through the ReLU and the head weights. The pooled-key gradient is
    accumulated without floating-point atomics. Gradients are returned for ``q``,
    ``weights`` and ``k_pooled`` (zeros when there is nothing to score), ``None`` for the
    rest. This Function is the kernel boundary for a fused implementation of the student.
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        weights: torch.Tensor,
        k_pooled: torch.Tensor,
        pool_ids_c: torch.Tensor,
        target: torch.Tensor,
        row_has_mass: torch.Tensor,
        row_count: Union[float, torch.Tensor],
        loss_coeff: float,
        hf_scoring: bool,
        use_relu: bool,
        chunk_bytes: int = _KPOOL_LOSS_CHUNK_BYTES,
    ) -> torch.Tensor:
        """Compute the loss in row chunks.

        Args:
            q: Indexer query ``[sq, batch, heads, dim]`` after any rotation.
            weights: Indexer head weights ``[sq, batch, heads]``.
            k_pooled: Pooled indexer keys ``[pools, batch, dim]`` after any rotation.
            pool_ids_c: Selected pool ids ``[batch, sq, slots]``, ``-1`` for unused slots.
            target: Teacher ``[batch, sq, slots]``, zero on unused slots.
            row_has_mass: Rows that count ``[batch, sq]`` (bool).
            row_count: Divisor of the summed KL (Python number or 0-dim tensor).
            loss_coeff: Indexer loss coefficient.
            hf_scoring: Whether the indexer uses HF KPool scoring.
            use_relu: Whether the indexer scores use ReLU.
            chunk_bytes: Byte budget of one row chunk.

        Returns:
            The scaled loss, a 0-dim FP32 tensor (FP64 when an input is FP64).
        """
        sq, batch, n_heads, head_dim = q.shape
        budget = pool_ids_c.size(-1)
        compute_dtype = _kpool_student_dtype(q, weights, k_pooled)
        row_mask = row_has_mass.to(dtype=torch.bool, device=q.device)
        kl_sum = torch.zeros((), dtype=compute_dtype, device=q.device)
        if sq > 0 and budget > 0:
            chunk = _kpool_loss_chunk_rows(
                sq, _kpool_student_bytes_per_row(batch, n_heads, head_dim, budget), chunk_bytes
            )
            for start in range(0, sq, chunk):
                end = min(start + chunk, sq)
                pool_ids = pool_ids_c[:, start:end]
                valid = pool_ids >= 0
                scores = _kpool_selected_pool_scores(
                    q[start:end],
                    weights[start:end],
                    k_pooled,
                    pool_ids,
                    hf_scoring=hf_scoring,
                    use_relu=use_relu,
                )
                log_probs = dsa_masking.masked_log_softmax(scores, valid)
                kl_per_row = dsa_indexer_loss.indexer_kl_per_row(
                    target[:, start:end].to(compute_dtype), log_probs, valid
                )
                kl_sum = kl_sum + kl_per_row.masked_fill(~row_mask[:, start:end], 0.0).sum()
        if isinstance(row_count, torch.Tensor):
            row_count = row_count.detach().to(dtype=compute_dtype, device=q.device)
            ctx.row_count = None
            saved_row_count = row_count
        else:
            ctx.row_count = float(row_count)
            saved_row_count = q.new_empty(0, dtype=compute_dtype)
        loss = kl_sum / row_count * loss_coeff
        ctx.save_for_backward(q, weights, k_pooled, pool_ids_c, target, row_mask, saved_row_count)
        ctx.loss_coeff = loss_coeff
        ctx.hf_scoring = hf_scoring
        ctx.use_relu = use_relu
        ctx.chunk_bytes = chunk_bytes
        return loss

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_loss: torch.Tensor):
        """Recompute the scores per chunk and return the q, weights and k_pooled gradients."""
        q, weights, k_pooled, pool_ids_c, target, row_mask, saved_row_count = ctx.saved_tensors
        row_count = ctx.row_count if ctx.row_count is not None else saved_row_count
        sq, batch, n_heads, head_dim = q.shape
        num_pools = k_pooled.size(0)
        budget = pool_ids_c.size(-1)
        compute_dtype = _kpool_student_dtype(q, weights, k_pooled)
        grad_q = torch.zeros(q.shape, dtype=compute_dtype, device=q.device)
        grad_weights = torch.zeros(weights.shape, dtype=compute_dtype, device=weights.device)
        grad_k = torch.zeros(k_pooled.shape, dtype=compute_dtype, device=k_pooled.device)
        if sq > 0 and budget > 0 and num_pools > 0:
            # d loss / d KL-sum, a 0-dim tensor: no host sync.
            scale = grad_loss.detach().to(dtype=compute_dtype) * ctx.loss_coeff / row_count
            chunk = _kpool_loss_chunk_rows(
                sq, _kpool_student_bytes_per_row(batch, n_heads, head_dim, budget), ctx.chunk_bytes
            )
            total_keys = num_pools * batch
            # Unused slots carry zero gradients; spread them over distinct dummy rows
            # because the deterministic index_add_ serializes duplicate ids.
            slots_per_chunk = chunk * batch * budget
            pad_rows = min(4096, slots_per_chunk)
            grad_k_ext = torch.zeros(
                (total_keys + pad_rows, head_dim), dtype=compute_dtype, device=k_pooled.device
            )
            pad_ids = total_keys + (
                torch.arange(slots_per_chunk, device=k_pooled.device, dtype=torch.int64) % pad_rows
            )
            batch_offsets = torch.arange(batch, device=q.device, dtype=torch.int64).view(
                batch, 1, 1
            )
            key_ids = None
            for start in range(0, sq, chunk):
                end = min(start + chunk, sq)
                pool_ids = pool_ids_c[:, start:end]
                valid = pool_ids >= 0
                head_scores, queries, keys = _kpool_selected_pool_head_scores(
                    q[start:end],
                    k_pooled,
                    pool_ids,
                    hf_scoring=ctx.hf_scoring,
                    use_relu=ctx.use_relu,
                    compute_dtype=compute_dtype,
                )
                scores = _kpool_reduce_selected_pool_head_scores(
                    head_scores, weights[start:end], hf_scoring=ctx.hf_scoring
                )
                probs = dsa_masking.masked_softmax(scores, valid)
                del scores
                chunk_target = target[:, start:end].to(compute_dtype).masked_fill(~valid, 0.0)
                # Exact logit gradient of KL(target || softmax): p * sum(target) - target.
                grad_scores = (
                    probs * chunk_target.sum(dim=-1, keepdim=True) - chunk_target
                ) * scale
                del probs, chunk_target
                grad_scores = grad_scores.masked_fill(
                    ~valid | ~row_mask[:, start:end].unsqueeze(-1), 0.0
                )
                with torch.autocast(device_type=q.device.type, enabled=False):
                    head_weights = weights[start:end].permute(1, 0, 2).to(compute_dtype)
                    # I = sum_h w_h * s_h  ->  dI/dw_h = s_h, dI/ds_h = w_h.
                    grad_weights[start:end] = (
                        torch.matmul(head_scores, grad_scores.unsqueeze(-1))
                        .squeeze(-1)
                        .permute(1, 0, 2)
                    )
                    grad_head_scores = head_weights.unsqueeze(-1) * grad_scores.unsqueeze(-2)
                    if ctx.use_relu:
                        grad_head_scores = grad_head_scores.masked_fill(head_scores <= 0, 0.0)
                    del head_scores
                    if ctx.hf_scoring:
                        grad_head_scores = grad_head_scores * (head_dim**-0.5)
                    grad_q[start:end] = torch.matmul(grad_head_scores, keys).permute(1, 0, 2, 3)
                    del keys
                    # [batch, rows, slots, dim]
                    grad_keys = torch.matmul(grad_head_scores.transpose(-1, -2), queries)
                    del grad_head_scores, queries
                n_slots = grad_keys.size(0) * grad_keys.size(1) * budget
                flat_ids = torch.where(
                    valid,
                    pool_ids.clamp_min(0).to(torch.int64) * batch + batch_offsets,
                    pad_ids[:n_slots].view_as(pool_ids),
                ).reshape(-1)
                key_ids = _kpool_accumulate_pool_key_grads(
                    grad_k_ext, flat_ids, grad_keys.reshape(-1, head_dim), key_ids
                )
                del grad_keys
            grad_k = grad_k_ext[:total_keys].view(num_pools, batch, head_dim)
        return (
            grad_q.to(q.dtype),
            grad_weights.to(weights.dtype),
            grad_k.to(k_pooled.dtype),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _kpool_pool_target_from_token_target(
    token_target: torch.Tensor, pool_ids: torch.Tensor, *, index_topk: int, pool_size: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pool-level teacher from a per-token-slot attention mass in the selector's layout.

    ``token_target`` ``[batch, sq, index_topk + tail]`` is the main attention's head-summed
    mass of each attended token slot, all on one per-row scale (pool slot ``j`` holds token
    slots ``j * pool_size .. j * pool_size + pool_size - 1``, the last columns are the tail),
    and ``pool_ids`` ``[batch, sq, index_topk // pool_size]`` the selected pools (``-1``
    unused). Each pool's mass is the sum of its token slots; the result follows
    ``_kpool_sparse_pool_target``: an exact-zero safe L1 normalization over the pools, rows
    without pool mass all zero, and ``p_E`` = tail mass / total mass (zero for rows without
    attended tokens).

    Returns:
        ``(target [batch, sq, pools] fp32, row_has_mass [batch, sq] bool,
        tail_mass [batch, sq] fp32)``.
    """
    budget = index_topk // pool_size
    batch, rows, width = token_target.shape
    if width < index_topk or tuple(pool_ids.shape) != (batch, rows, budget):
        raise ValueError(
            f"KPool token target [{batch}, {rows}, >= {index_topk}] and pool ids "
            f"[{batch}, {rows}, {budget}] expected, got {tuple(token_target.shape)} and "
            f"{tuple(pool_ids.shape)}."
        )
    token_target = token_target.float()
    pool_mass = token_target[..., :index_topk].reshape(batch, rows, budget, pool_size).sum(-1)
    pool_mass = pool_mass.masked_fill(pool_ids < 0, 0.0)
    target = _normalize_teacher_target_exact_zero(pool_mass)
    row_has_mass = pool_mass.sum(dim=-1) > 0
    total_mass = token_target.sum(dim=-1)
    tail_mass = token_target[..., index_topk:].sum(dim=-1) / torch.where(
        total_mass > 0, total_mass, torch.ones_like(total_mass)
    )
    return target, row_has_mass, tail_mass


def _kpool_loss_rows(
    row_has_mass: torch.Tensor,
    tail_mass: torch.Tensor,
    *,
    query_valid_rows: Optional[torch.Tensor],
    calculate_per_token_loss: bool,
    layer_number: int,
    num_layers: int,
    reduce_group: Optional[torch.distributed.ProcessGroup],
    avg_group: Optional[torch.distributed.ProcessGroup],
) -> Tuple[torch.Tensor, Union[float, torch.Tensor]]:
    """The rows of a KPool sparse loss and its divisor; records the layer's tail mass ``p_E``.

    The rows are those with pool mass, and only the valid query rows when
    ``query_valid_rows`` is given. The divisor is the token loss's: the valid row count, or
    every row, or 1 under ``calculate_per_token_loss``. ``p_E`` goes to the metric tracker as
    its sum over the loss's rows and their number (device-side sums: no host sync); the
    other arguments are the indexer loss's tracker arguments.

    Returns:
        ``(row_mask [batch, sq] bool, row_count)``.
    """
    batch, rows = row_has_mass.shape
    row_mask = row_has_mass
    if query_valid_rows is not None:
        row_mask = row_mask & query_valid_rows.to(dtype=torch.bool, device=row_mask.device)
    # Rows without pool mass (p_E = 1) and padded rows stay out of the p_E mean, as they
    # stay out of the loss.
    row_weight = row_mask.to(torch.float32)
    DSAIndexerLossLoggingHelper.save_tail_mass_to_tracker(
        tail_mass_sum=(tail_mass.float() * row_weight).sum(),
        row_count=row_weight.sum(),
        layer_number=layer_number,
        num_layers=num_layers,
        reduce_group=reduce_group,
        avg_group=avg_group,
    )
    if calculate_per_token_loss:
        row_count = 1.0
    elif query_valid_rows is not None:
        row_count = query_valid_rows.sum().to(torch.float32).clamp_min(1.0)
    else:
        row_count = float(max(batch * rows, 1))
    return row_mask, row_count


def _kpool_scoring_inputs(
    q: torch.Tensor, k_pooled: torch.Tensor, rotate: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``q`` and the pooled keys as the selector scores them: Hadamard-rotated when enabled."""
    if rotate:
        if q.numel():
            q = rotate_activation(q)
        if k_pooled.numel():
            k_pooled = rotate_activation(k_pooled)
    return q, k_pooled


class DSAIndexerLossAutoScaler(torch.autograd.Function):
    """An AutoScaler that triggers the backward pass and scales the grad for indexer loss.

    This custom autograd function attaches a KL divergence loss to the activation
    to train the indexer to predict attention scores without affecting the forward pass.
    """

    main_loss_backward_scale: Optional[torch.Tensor] = None

    @staticmethod
    def forward(ctx, output: torch.Tensor, indexer_loss: torch.Tensor):
        """Preserve the indexer_loss by storing it in the context to avoid garbage collection.

        Args:
            output: The output tensor (activation).
            indexer_loss: The indexer KL divergence loss tensor.

        Returns:
            torch.Tensor: The output tensor unchanged.
        """
        ctx.save_for_backward(indexer_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Compute and scale the gradient for indexer loss.

        Args:
            grad_output: The gradient of the output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The gradient of the output, scaled indexer loss
                gradient.
        """
        (indexer_loss,) = ctx.saved_tensors
        if DSAIndexerLossAutoScaler.main_loss_backward_scale is None:
            DSAIndexerLossAutoScaler.main_loss_backward_scale = torch.tensor(
                1.0, device=indexer_loss.device
            )
        indexer_loss_backward_scale = DSAIndexerLossAutoScaler.main_loss_backward_scale.to(
            device=indexer_loss.device
        )
        scaled_indexer_loss_grad = torch.ones_like(indexer_loss) * indexer_loss_backward_scale
        return grad_output, scaled_indexer_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        """Set the scale of the indexer loss.

        Args:
            scale: The scale value to set.
        """
        if not isinstance(scale, torch.Tensor):
            raise TypeError("DSAIndexerLossAutoScaler.set_loss_scale requires a torch.Tensor.")
        scale = scale.detach()

        if DSAIndexerLossAutoScaler.main_loss_backward_scale is None:
            DSAIndexerLossAutoScaler.main_loss_backward_scale = scale
        else:
            DSAIndexerLossAutoScaler.main_loss_backward_scale.copy_(scale)


@dataclass
class DSAIndexerSubmodules:
    """
    Configuration class for specifying the submodules of an DSA Indexer.

    Args:
        linear_wq_b: Linear projection for query bottleneck expansion.
        linear_wk: Linear projection for key.
        k_norm: Layer normalization for key.
        linear_weights_proj: Linear projection for attention weights.
    """

    linear_wq_b: Union[ModuleSpec, type] = None
    linear_wk: Union[ModuleSpec, type] = None
    k_norm: Union[ModuleSpec, type] = None
    linear_weights_proj: Union[ModuleSpec, type] = None


@dataclass
class DSAttentionSubmodules:
    """
    Configuration class for specifying the submodules of DSAttention.

    Args:
        indexer: DSA Indexer module for computing sparse attention indices.
    """

    indexer: Union[ModuleSpec, type] = None


def _dsa_weights_proj_te_gemm_is_unsupported(error: RuntimeError) -> bool:
    """Return whether a TE GEMM error permits the documented FP32 fallback."""
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "cublas_status_not_supported",
            "unable to find suitable cublas gemm algorithm",
            "unable to find any suitable algorithms",
        )
    )


def _warn_dsa_weights_proj_te_gemm_fallback(error: RuntimeError) -> None:
    """Warn once when TE cannot provide the requested FP32 GEMM output."""
    global _DSA_WEIGHTS_PROJ_TE_GEMM_FALLBACK_WARNED
    if _DSA_WEIGHTS_PROJ_TE_GEMM_FALLBACK_WARNED:
        return
    _DSA_WEIGHTS_PROJ_TE_GEMM_FALLBACK_WARNED = True
    log_single_rank(
        logger,
        logging.WARNING,
        "Transformer Engine GEMM does not support BF16-input/FP32-output for the DSA indexer "
        "weights projection on this platform; using torch.mm with FP32 operands instead. "
        f"Original TE error: {error}",
    )


def _warn_dsa_checkpoint_topk_reuse_fallback(layer_number: int, reason: str) -> None:
    """Warn once when a checkpoint replay cannot reuse its first call's top-k selection."""
    global _DSA_CHECKPOINT_TOPK_REUSE_FALLBACK_WARNED
    if _DSA_CHECKPOINT_TOPK_REUSE_FALLBACK_WARNED:
        return
    _DSA_CHECKPOINT_TOPK_REUSE_FALLBACK_WARNED = True
    # Logged on every rank: a fallback can be specific to one pipeline stage.
    logger.warning(
        "dsa_indexer_reuse_checkpoint_topk is enabled, but the checkpoint replay of DSA layer "
        f"{layer_number} reran the dense indexer and top-k: {reason}. Further occurrences are "
        "not logged."
    )


_KPOOL_SCORER_FALLBACK_REASONS_WARNED: set = set()


def _warn_kpool_scorer_fallback_once(reason: str) -> None:
    """Warn once per reason when ``dsa_indexer_kpool_scorer_backend='cudnn'`` cannot apply."""
    if reason in _KPOOL_SCORER_FALLBACK_REASONS_WARNED:
        return
    _KPOOL_SCORER_FALLBACK_REASONS_WARNED.add(reason)
    logger.warning(
        "dsa_indexer_kpool_scorer_backend='cudnn' keeps the PyTorch KPool scorer for this "
        f"call: {reason}. Further calls with this reason are not logged."
    )


_KPOOL_LOSS_PATHS_LOGGED: set = set()


def _log_kpool_loss_path_once(config: TransformerConfig, pytorch_reason: Optional[str]) -> None:
    """Log once per reason which implementation runs the trainable KPool indexer loss.

    ``pytorch_reason`` is None when the cuDNN frontend kernels run it, otherwise why the
    PyTorch ``KPoolSparseIndexerLoss`` runs. That is a warning when the configuration asks
    for the cuDNN kernels and the cuDNN KPool scorer, and information otherwise.
    """
    if pytorch_reason in _KPOOL_LOSS_PATHS_LOGGED:
        return
    _KPOOL_LOSS_PATHS_LOGGED.add(pytorch_reason)
    if pytorch_reason is None:
        logger.info(
            "The trainable KPool indexer loss runs on the cuDNN frontend kernels (score "
            "recompute teacher, cuDNN KPool scorer student, sparse indexer backward)."
        )
        return
    asks_for_cudnn = (
        config.dsa_kernel_backend == "cudnn" and config.dsa_indexer_kpool_scorer_backend == "cudnn"
    )
    logger.log(
        logging.WARNING if asks_for_cudnn else logging.INFO,
        "The trainable KPool indexer loss runs in PyTorch (KPoolSparseIndexerLoss), not on "
        f"the cuDNN frontend kernels: {pytorch_reason}. Further forwards with this reason "
        "are not logged.",
    )


def _reject_mxfp8_indexer_reference_fallback(
    config: TransformerConfig, what: str, reason: str, path: str = "the BF16 reference path"
) -> None:
    """Raise instead of scoring the indexer without MXFP8 when MXFP8 was requested.

    Only the cuDNN fused indexer hooks honour ``dsa_indexer_precision='mxfp8'``. The
    PyTorch reference scorer is BF16 and KPool selection has its own scorer, so taking
    either would report other numerics and speed for a run configured as MXFP8.
    """
    if getattr(config, "dsa_indexer_precision", "bf16") != "mxfp8":
        return
    raise RuntimeError(
        "dsa_indexer_precision='mxfp8' requires the cuDNN fused DSA indexer, but "
        f"{what} would be computed by {path}: {reason}. "
        "Use dsa_indexer_precision='bf16' for this layout."
    )


def _dsa_weights_proj_forward_gemm(
    x: torch.Tensor, weight: torch.Tensor, te_gemm_supported: Optional[bool]
) -> Tuple[torch.Tensor, Optional[bool]]:
    """Project BF16 indexer weights with FP32 accumulation and output."""
    x_shape = x.shape
    x_2d = x.reshape(-1, x_shape[-1])

    if te_gemm_supported is not False and te_general_gemm is not None and x_2d.is_cuda:
        try:
            output = te_general_gemm(weight, x_2d, out_dtype=torch.float32, layout="TN")[0]
            return output.reshape(*x_shape[:-1], weight.size(0)), True
        except RuntimeError as error:
            if not _dsa_weights_proj_te_gemm_is_unsupported(error):
                raise
            te_gemm_supported = False
            _warn_dsa_weights_proj_te_gemm_fallback(error)

    # Do not emulate the requested precision by casting a low-precision output.
    # Casting both operands makes the fallback a genuine FP32 linear operation.
    output = torch.mm(x_2d.float(), weight.float().t())
    return output.reshape(*x_shape[:-1], weight.size(0)), te_gemm_supported


def _dsa_weights_proj_wgrad(
    x: torch.Tensor,
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    linear: torch.nn.Module,
    is_first_microbatch: Optional[bool],
) -> Tuple[torch.Tensor, None]:
    """Compute or accumulate the high-precision weight gradient."""
    x_2d = x.reshape(-1, x.size(-1))
    grad_output_2d = grad_output.reshape(-1, grad_output.size(-1))
    grad_weight = torch.mm(grad_output_2d.float().t(), x_2d.float())

    if getattr(linear, "fuse_wgrad_accumulation", False):
        if hasattr(weight, "__fsdp_param__"):
            main_grad = weight.get_main_grad()
        else:
            main_grad = getattr(weight, "main_grad", None)
        if main_grad is not None:
            if is_first_microbatch is True or getattr(weight, "overwrite_main_grad", False):
                main_grad.copy_(grad_weight)
            else:
                main_grad.add_(grad_weight)
            weight.main_grad = main_grad
            if hasattr(weight, "grad_added_to_main_grad"):
                weight.grad_added_to_main_grad = True
            # The delayed-WGrad owner ignores this return value when accumulation is fused.
            return grad_weight, None

    return grad_weight.to(dtype=weight.dtype), None


class _DSAWeightsProjection(torch.autograd.Function):
    """BF16-operand, FP32-output linear used only by the DSA indexer."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        linear: torch.nn.Module,
        indexer: torch.nn.Module,
    ) -> torch.Tensor:
        """Run the FP32-output projection and save operands for backward."""
        ctx.save_for_backward(x, weight)
        ctx.linear = linear
        # This path does not use TE's parameter-transpose cache. Keep the
        # first-microbatch signal solely for fused WGrad overwrite/accumulate semantics.
        ctx.is_first_microbatch = getattr(linear, "is_first_microbatch", None)
        if hasattr(linear, "is_first_microbatch"):
            linear.is_first_microbatch = False
        output, indexer._weights_proj_te_gemm_supported = _dsa_weights_proj_forward_gemm(
            x, weight, indexer._weights_proj_te_gemm_supported
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Propagate gradients with MCore's deferred and fused-WGrad semantics."""
        x, weight = ctx.saved_tensors
        linear = ctx.linear

        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_output_2d = grad_output.reshape(-1, grad_output.size(-1))
            grad_input = torch.mm(grad_output_2d.float(), weight.float())
            grad_input = grad_input.reshape_as(x).to(dtype=x.dtype)

        grad_weight = None
        if ctx.needs_input_grad[1]:
            if (
                getattr(linear, "wgrad_store", None) is not None
                and linear.wgrad_store.delay_wgrad_compute()
            ):
                wgrad_func = functools.partial(
                    _dsa_weights_proj_wgrad,
                    weight=weight,
                    linear=linear,
                    is_first_microbatch=ctx.is_first_microbatch,
                )
                linear.wgrad_store.put([x, grad_output], wgrad_func)
            else:
                grad_weight, _ = _dsa_weights_proj_wgrad(
                    x, grad_output, weight, linear, ctx.is_first_microbatch
                )

            if getattr(linear, "fuse_wgrad_accumulation", False) and hasattr(
                weight, "grad_added_to_main_grad"
            ):
                # Keep a dummy grad for MCore's parameter hook, including delayed WGrad.
                zero_dummy = getattr(weight, "zero_out_wgrad", False)
                if get_dummy_wgrad is not None:
                    grad_weight = get_dummy_wgrad(list(weight.shape), weight.dtype, zero=zero_dummy)
                elif zero_dummy:
                    grad_weight = torch.zeros_like(weight, requires_grad=False)
                else:
                    grad_weight = torch.empty_like(weight, requires_grad=False)

        return grad_input, grad_weight, None, None


def _dsa_weights_projection_fp32(x: torch.Tensor, indexer: torch.nn.Module) -> torch.Tensor:
    """Apply the non-quantized DSA indexer projection with a true FP32 output."""
    linear = indexer.linear_weights_proj
    weight = getattr(linear, "weight", None)
    if weight is None:
        raise RuntimeError("DSA indexer linear_weights_proj must expose a weight parameter.")
    ensure_params_ready([weight])
    return _DSAWeightsProjection.apply(x, weight, linear, indexer)


def _validate_dsa_kpool_config(config: TransformerConfig) -> None:
    """Validate KPool invariants for direct Python API construction as well as config parsing."""
    pool_size = getattr(config, "dsa_indexer_kpool", 1)
    if pool_size < 1:
        raise ValueError(f"dsa_indexer_kpool must be positive, got {pool_size}.")
    if pool_size == 1:
        return

    index_topk = getattr(config, "dsa_indexer_topk", None)
    if index_topk is None:
        raise ValueError("dsa_indexer_topk must be set when dsa_indexer_kpool is greater than 1.")
    if index_topk < 1:
        raise ValueError(
            "dsa_indexer_topk must be positive when dsa_indexer_kpool is greater than 1."
        )
    if index_topk % pool_size != 0:
        raise ValueError(
            "dsa_indexer_topk must be divisible by dsa_indexer_kpool for pool selection; "
            f"got topk={index_topk}, kpool={pool_size}."
        )
    # The same rules as TransformerConfig, for configs built without its validation.
    # Attributes a minimal config lacks read as their defaults.
    if getattr(config, "dsa_indexer_mode", "standard") == "simplified":
        raise ValueError(
            "dsa_indexer_mode='simplified' does not support KPool selection "
            "(dsa_indexer_kpool > 1): the simplified indexer has no pool compression, so the "
            f"pools would be silently ignored; got dsa_indexer_kpool={pool_size}."
        )
    if getattr(config, "dsa_indexer_reuse_checkpoint_topk", False):
        raise ValueError(
            "dsa_indexer_reuse_checkpoint_topk does not support KPool selection "
            f"(dsa_indexer_kpool > 1), got dsa_indexer_kpool={pool_size}."
        )
    if (getattr(config, "dsa_indexer_loss_coeff", None) or 0.0) > 0:
        if not getattr(config, "dsa_indexer_use_sparse_loss", False):
            raise ValueError(
                "DSA indexer loss is not supported with kpool selection unless it is the "
                "sparse pool-level loss: set dsa_indexer_use_sparse_loss=True to train the "
                "KPool indexer, dsa_indexer_loss_coeff=0 to keep it frozen, or "
                "dsa_indexer_kpool=1."
            )
        if getattr(config, "dsa_indexer_kpool_use_quantization", False):
            raise ValueError(
                "A trainable KPool indexer (dsa_indexer_kpool > 1 with dsa_indexer_loss_coeff "
                "> 0) does not support FP8 KPool scoring (dsa_indexer_kpool_use_quantization): "
                "its E4M3 round trip has no defined training gradient."
            )
        if getattr(config, "cuda_graph_impl", "none") == "full_iteration":
            raise ValueError(
                "A trainable KPool indexer (dsa_indexer_kpool > 1 with dsa_indexer_loss_coeff "
                "> 0) does not support cuda_graph_impl='full_iteration'."
            )
        replay = _trainable_kpool_replay(config)
        if replay is not None:
            raise ValueError(_trainable_kpool_replay_message(replay))


def _trainable_kpool_replay(config) -> Optional[str]:
    """Name the activation recompute that would replay a DSA layer's selection, if any."""
    granularity = getattr(config, "recompute_granularity", None)
    if granularity == "full":
        return "recompute_granularity='full'"
    if granularity == "selective" and "core_attn" in (
        getattr(config, "recompute_modules", None) or []
    ):
        return "selective recompute of 'core_attn'"
    return None


def _trainable_kpool_replay_message(replay: str) -> str:
    """Rejection message for a trainable KPool indexer under a replaying recompute."""
    return (
        "A trainable KPool indexer (dsa_indexer_kpool > 1 with dsa_indexer_loss_coeff > 0) "
        f"does not support {replay}: the checkpoint replay would select the pools again, and "
        "the cuDNN top-k does not repeat its choice among tied scores, so the replayed sparse "
        "attention and indexer loss could use other pools than the first call, whose output "
        "the forward used. Use a recompute that does not replay core attention (e.g. "
        "selective 'mla_up_proj' or 'mhc')."
    )


class DSAIndexer(MegatronModule):
    """
    DSA Lightning Indexer for DeepSeek Sparse Attention.

    Computes index scores to identify the top-k most relevant key-value pairs for each query in
    sparse attention.

    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L431-L480
    """

    def _projection_quantization_context(self, *, is_init: bool = False):
        """Use the enclosing quantization context only when explicitly enabled."""
        if not self.config.dsa_indexer_qk_proj_use_quantization:
            return get_fp8_disabled_context(self.config, is_init=is_init)
        return nullcontext()

    def __init__(
        self,
        config: TransformerConfig,
        submodules: DSAIndexerSubmodules,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        """Initialize the indexer.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
            submodules (DSAIndexerSubmodules): Indexer submodules specification.
            pg_collection (ProcessGroupCollection, optional): Process groups for the indexer.
        """
        super().__init__(config=config)
        _validate_dsa_kpool_config(config)
        self.hidden_size = self.config.hidden_size
        self.qk_pos_emb_head_dim = self.config.qk_pos_emb_head_dim
        self.q_lora_rank = (
            self.config.q_lora_rank
            if self.config.q_lora_rank is not None
            else self.config.hidden_size
        )

        self.index_n_heads = self.config.dsa_indexer_n_heads
        self.index_head_dim = self.config.dsa_indexer_head_dim
        self.index_topk = self.config.dsa_indexer_topk

        self.softmax_scale: float = self.index_head_dim**-0.5

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        self.pg_collection = pg_collection

        # Initialize Position Embedding.
        # NoPE (qk_pos_emb_head_dim == 0, e.g. GLM-5.3-Flash indexer): skip the
        # rotary embedding entirely; constructing it with dim=0 yields empty
        # inv_freq / NaN, and the forward path skips RoPE split/apply.
        self.use_rope = self.qk_pos_emb_head_dim > 0
        if not self.use_rope:
            self.rotary_pos_emb = None
        elif self.config.rope_type == 'rope':
            self.rotary_pos_emb = RotaryEmbedding(
                self.qk_pos_emb_head_dim,
                rotary_percent=self.config.rotary_percent,
                rotary_base=self.config.rotary_base,
                cp_group=self.pg_collection.cp,
            )
        elif self.config.rope_type == 'yarn':
            self.rotary_pos_emb = YarnRotaryEmbedding(
                self.qk_pos_emb_head_dim,
                rotary_base=self.config.rotary_base,
                scaling_factor=self.config.rotary_scaling_factor,
                original_max_position_embeddings=self.config.original_max_position_embeddings,
                beta_fast=self.config.beta_fast,
                beta_slow=self.config.beta_slow,
                mscale=self.config.mscale,
                mscale_all_dim=self.config.mscale_all_dim,
                cp_group=self.pg_collection.cp,
            )
        else:
            raise ValueError(
                f'Unsupported RoPE type: {self.config.rope_type}, supported types are "rope" and '
                f'"yarn"'
            )

        # Query/key projections follow the enclosing quantization policy unless opted out.
        with self._projection_quantization_context(is_init=True):
            self.linear_wq_b = build_module(
                submodules.linear_wq_b,
                self.q_lora_rank,
                self.index_n_heads * self.index_head_dim,
                config=self.config,
                init_method=self.config.init_method,
                bias=False,
                skip_bias_add=False,
                skip_weight_param_allocation=False,
                parallel_mode="duplicated",
            )
            self.linear_wk = build_module(
                submodules.linear_wk,
                self.hidden_size,
                self.index_head_dim,
                config=self.config,
                init_method=self.config.init_method,
                bias=False,
                skip_bias_add=False,
                skip_weight_param_allocation=False,
                parallel_mode="duplicated",
            )

        k_norm_config = copy.copy(self.config)
        k_norm_config.normalization = "LayerNorm"
        k_norm_eps = (
            self.config.dsa_indexer_k_norm_epsilon
            if self.config.dsa_indexer_k_norm_epsilon is not None
            else self.config.layernorm_epsilon
        )
        self.k_norm = build_module(
            submodules.k_norm, config=k_norm_config, hidden_size=self.index_head_dim, eps=k_norm_eps
        )

        weights_proj_init_context = (
            get_fp8_disabled_context(self.config, is_init=True)
            if not self.config.dsa_indexer_weights_proj_use_quantization
            else nullcontext()
        )
        with weights_proj_init_context:
            self.linear_weights_proj = build_module(
                submodules.linear_weights_proj,
                self.hidden_size,
                self.index_n_heads,
                config=self.config,
                init_method=self.config.init_method,
                bias=False,
                skip_bias_add=False,
                skip_weight_param_allocation=False,
                parallel_mode="duplicated",
            )
        self._weights_proj_te_gemm_supported: Optional[bool] = None

        # Pool compression uses replicated per-token gates and slot-position biases.
        self.index_kpool = self.config.dsa_indexer_kpool
        self.index_kpool_use_quantization = self.config.dsa_indexer_kpool_use_quantization
        self.index_kpool_always_select_tail = self.config.dsa_indexer_kpool_always_select_tail
        # Per-token gate score of a frozen KPool selector; set in forward_before_topk.
        self._kpool_gate_score: Optional[torch.Tensor] = None
        # A positive loss coefficient trains the whole KPool indexer, the pool gate and the
        # slot bias included, with the sparse pool-level KL (KPoolSparseIndexerLoss);
        # otherwise the KPool selector is frozen.
        self._kpool_trainable = (
            self.index_kpool > 1 and (self.config.dsa_indexer_loss_coeff or 0.0) > 0
        )
        if self.index_kpool > 1:
            # fp32 [kpool, index_head_dim] additive positional bias per pool slot.
            kpool_param_device = (
                "cpu" if self.config.use_cpu_initialization else torch.cuda.current_device()
            )
            self.index_kpool_compress_ape = mark_keep_in_fp32(
                torch.nn.Parameter(
                    torch.zeros(
                        self.index_kpool,
                        self.index_head_dim,
                        dtype=torch.float32,
                        device=kpool_param_device,
                    ),
                    requires_grad=self._kpool_trainable,
                )
            )
            # A bias-like per-slot vector, not a weight matrix: keep it out of Muon.
            setattr(self.index_kpool_compress_ape, "use_muon", False)
            # [index_head_dim, hidden_size]; gate_score = F.linear(x, gate) = x @ gate^T
            # -> [seqlen, index_head_dim]. Matches vLLM's checkpoint name (no .weight suffix).
            self.index_kpool_compress_gate = torch.nn.Parameter(
                torch.empty(
                    self.index_head_dim,
                    self.hidden_size,
                    dtype=self.config.params_dtype,
                    device=kpool_param_device,
                ),
                requires_grad=self._kpool_trainable,
            )
            nn.init.normal_(self.index_kpool_compress_gate, std=0.01)
        else:
            self.index_kpool_compress_ape = None
            self.index_kpool_compress_gate = None

        # Indexer projections are duplicated across tensor-parallel ranks, so their gradients
        # should be averaged during final gradient synchronization.
        for param in self.parameters():
            setattr(param, "average_gradients_across_tp_domain", True)

        # The sparse-attention top-k path is discrete and runs under ``no_grad``.  Without the
        # auxiliary indexer loss, none of these parameters can receive a gradient; leave them
        # out of DDP/optimizer bookkeeping instead of failing on an unused-parameter reset.
        if (self.config.dsa_indexer_loss_coeff or 0.0) <= 0:
            for param in self.parameters():
                param.requires_grad_(False)

    def _project_indexer_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Run ``linear_weights_proj`` according to the DSA-specific precision contract."""
        weights_proj_context = (
            get_fp8_disabled_context(self.config)
            if not self.config.dsa_indexer_weights_proj_use_quantization
            else nullcontext()
        )
        with weights_proj_context:
            if self.config.dsa_indexer_weights_proj_output_dtype == "fp32":
                return _dsa_weights_projection_fp32(x, self)

            weights, _ = self.linear_weights_proj(x)
            return weights.to(dtype=torch.bfloat16)

    def backward_dw(self):
        """Compute the deferred weight gradients of the indexer projections."""
        self.linear_wq_b.backward_dw()
        self.linear_wk.backward_dw()
        self.linear_weights_proj.backward_dw()

    def _apply_rope(
        self,
        x: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor],
        mscale: float,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        rotary_pos_cos: Optional[torch.Tensor] = None,
        rotary_pos_sin: Optional[torch.Tensor] = None,
    ):
        """Apply RoPE to the input tensor."""
        # NoPE: no positional component; x is all nope, return unchanged.
        if not self.use_rope:
            return x
        if rotary_pos_cos is not None or rotary_pos_sin is not None:
            assert rotary_pos_cos is not None and rotary_pos_sin is not None
            assert fused_mla_rope_inplace is not None, "Fused MLA RoPE is not available"
            if cu_seqlens is not None and cu_seqlens.device != x.device:
                cu_seqlens = cu_seqlens.to(device=x.device)
            squeezed_batch_dim = False
            # THD RoPE expects [t, h, d], while indexer tensors are [t, 1, h, d].
            if cu_seqlens is not None and x.ndim == 4 and x.size(1) == 1:
                x = x.squeeze(1)
                squeezed_batch_dim = True
            x = fused_mla_rope_inplace(
                x,
                rotary_pos_cos,
                rotary_pos_sin,
                nope_dim=self.index_head_dim - self.qk_pos_emb_head_dim,
                emb_dim=self.qk_pos_emb_head_dim,
                cu_seqlens_q=cu_seqlens,
                cp_rank=self.pg_collection.cp.rank(),
                cp_size=self.pg_collection.cp.size(),
                rope_first=True,
            )
            if squeezed_batch_dim:
                x = x.unsqueeze(1)
            return x

        assert rotary_pos_emb is not None
        # x_pe   [seqlen, batch, *, qk_pos_emb_head_dim]
        # x_nope [seqlen, batch, *, index_head_dim - qk_pos_emb_head_dim]
        # To align with DeepSeek's implementation,
        # x_pe is placed at the front, and x_nope is placed at the back.
        x_pe, x_nope = torch.split(
            x, [self.qk_pos_emb_head_dim, self.index_head_dim - self.qk_pos_emb_head_dim], dim=-1
        )
        squeezed_batch_dim = False
        if cu_seqlens is not None and cu_seqlens.device != x_pe.device:
            cu_seqlens = cu_seqlens.to(device=x_pe.device)
        # THD RoPE path expects [t, h, d], while indexer tensors are [t, 1, h, d].
        if cu_seqlens is not None and x_pe.ndim == 4 and x_pe.size(1) == 1:
            x_pe = x_pe.squeeze(1)
            squeezed_batch_dim = True
        x_pe = apply_rotary_pos_emb(
            x_pe,
            rotary_pos_emb,
            config=self.config,
            cu_seqlens=cu_seqlens,
            mscale=mscale,
            cp_group=self.pg_collection.cp,
            # This flag is for the MLA-style interleaving in RoPE.
            mla_rotary_interleaved=self.config.dsa_indexer_rope_interleaved,
            max_seqlen=max_seqlen,
        )
        if squeezed_batch_dim:
            x_pe = x_pe.unsqueeze(1)
        # [seqlen, batch, *, index_head_dim]
        x = torch.cat([x_pe, x_nope], dim=-1)
        return x

    def kpool_pool_keys(
        self,
        k_local: torch.Tensor,
        gate_local: Optional[torch.Tensor],
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress indexer keys into KPool pools under the caller's grad mode.

        ``k_local`` and ``gate_local`` are ``[tokens, batch, index_head_dim]``. Without
        ``cu_seqlens`` the complete pools of the token axis are compressed; with packed
        ``cu_seqlens`` the pools restart at every document, as ``fused_qk_topk_kpool``
        compresses them. Uses the configured compression profile, before any rotation.

        Returns:
            ``(k_pooled [pools, batch, index_head_dim], pool_token_base [pools])``, where
            ``pool_token_base`` holds each pool's first token on the given token axis
            (``-1`` for an invalid packed capacity pool).
        """
        key_dtype_compression = self.config.dsa_indexer_kpool_key_dtype_compression
        pool_size = self.index_kpool
        if cu_seqlens is not None and cu_seqlens.numel() >= 2:
            if k_local.size(0) < pool_size:
                # No pool fits; still compress (nothing) through the gate and the slot bias,
                # so they stay in this rank's autograd graph and receive a zero gradient.
                return (
                    _kpool_compress_keys(
                        k_local[:0],
                        gate_local[:0] if gate_local is not None else None,
                        self.index_kpool_compress_ape,
                        pool_size,
                        kpool_key_dtype_compression=key_dtype_compression,
                    ),
                    torch.empty(0, device=k_local.device, dtype=torch.int64),
                )
            return _kpool_compress_keys_per_seg(
                k_local,
                gate_local,
                self.index_kpool_compress_ape,
                pool_size,
                cu_seqlens,
                kpool_key_dtype_compression=key_dtype_compression,
            )
        k_pooled = _kpool_compress_keys(
            k_local,
            gate_local,
            self.index_kpool_compress_ape,
            pool_size,
            kpool_key_dtype_compression=key_dtype_compression,
        )
        pool_token_base = (
            torch.arange(k_pooled.size(0), device=k_local.device, dtype=torch.int64) * pool_size
        )
        return k_pooled, pool_token_base

    def forward_before_topk(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        packed_seq_params: Optional[PackedSeqParams] = None,
        return_kpool_gate_score: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """All computations before topk.

        Returns ``(q, k, weights)``. A KPool indexer also computes the per-token pool-gate
        scores: by default they are stashed on the module for the subsequent selection
        call (the frozen selector's contract); with ``return_kpool_gate_score`` they are
        returned as a fourth value instead and nothing is stashed (the trainable path).
        """
        packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"

        # =========================================
        # Prepare RoPE params
        # =========================================
        rotary_pos_cos = rotary_pos_sin = None
        if self.use_rope:
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                None, None, x, self.config, packed_seq_params
            )
            fused_indexer_rope = (
                self.config.dsa_indexer_rope_interleaved and should_use_fused_mla_rope(self.config)
            )
            if fused_indexer_rope:
                rotary_pos_cos, rotary_pos_sin = self.rotary_pos_emb.get_cached_cos_sin(
                    rotary_seq_len, dtype=x.dtype, packed_seq=packed_seq
                )
                rotary_pos_emb = None
                mscale = 1.0
            elif self.config.rope_type == "rope":
                rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)
                mscale = 1.0
            else:
                rotary_pos_emb, mscale = self.rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)
        else:
            # NoPE: no rotary embedding; _apply_rope is a no-op.
            rotary_pos_emb = None
            mscale = 1.0
        if packed_seq:
            cu_seqlens_q, cu_seqlens_kv = dsa_layout.get_packed_qk_cu_seqlens(packed_seq_params)
            max_seqlen_q = packed_seq_params.max_seqlen_q
            max_seqlen_kv = packed_seq_params.max_seqlen_kv
        else:
            cu_seqlens_q = cu_seqlens_kv = None
            max_seqlen_q = max_seqlen_kv = None

        # =========================================
        # Gather inputs if sp is enabled
        # =========================================
        if self.config.sequence_parallel and self.pg_collection.tp.size() > 1:
            x = gather_from_sequence_parallel_region(x, group=self.pg_collection.tp)
            qr = gather_from_sequence_parallel_region(qr, group=self.pg_collection.tp)

        # =========================================
        # Get sequence length and batch size
        # =========================================
        seqlen, bsz, _ = x.size()

        # =========================================
        # q linear and apply rope to q
        # =========================================
        # [seqlen, batch, q_lora_rank] -> [seqlen, batch, index_n_heads * index_head_dim]
        with self._projection_quantization_context():
            q, _ = self.linear_wq_b(qr)
        # [seqlen, batch, index_n_heads * index_head_dim]
        #   -> [seqlen, batch, index_n_heads, index_head_dim]
        q = q.reshape(seqlen, bsz, self.index_n_heads, self.index_head_dim)
        q = self._apply_rope(
            q,
            rotary_pos_emb,
            mscale,
            cu_seqlens=cu_seqlens_q,
            max_seqlen=max_seqlen_q,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
        )
        if self.config.dsa_indexer_rotate_activation and self.index_kpool <= 1:
            q = rotate_activation(q)

        # =========================================
        # Key projection, normalization, and rotation follow the configured precision policy.
        # =========================================
        with self._projection_quantization_context():
            # [seqlen, batch, hidden_size] -> [seqlen, batch, index_head_dim]
            k, _ = self.linear_wk(x)
            if self.config.dsa_indexer_k_norm_fp32:
                k_dtype = k.dtype
                # TE LayerNorm casts its input to the parameter dtype, so passing
                # k.float() to the BF16 module does not guarantee FP32 math.
                ensure_params_ready([self.k_norm.weight, self.k_norm.bias])
                k_norm_weight = self.k_norm.weight.float()
                if self.config.layernorm_zero_centered_gamma:
                    k_norm_weight = k_norm_weight + 1.0
                with torch.autocast(device_type=k.device.type, enabled=False):
                    k = torch.nn.functional.layer_norm(
                        k.float(),
                        (self.index_head_dim,),
                        k_norm_weight,
                        self.k_norm.bias.float(),
                        self.k_norm.eps,
                    ).to(dtype=k_dtype)
            else:
                k = self.k_norm(k)
            # [seqlen, batch, index_head_dim] -> [seqlen, batch, 1, index_head_dim]
            k = k.reshape(seqlen, bsz, 1, self.index_head_dim)
            k = self._apply_rope(
                k,
                rotary_pos_emb,
                mscale,
                cu_seqlens=cu_seqlens_kv,
                max_seqlen=max_seqlen_kv,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
            )
            # [seqlen, batch, 1, index_head_dim] -> [seqlen, batch, index_head_dim]
            k = k.reshape(seqlen, bsz, self.index_head_dim)

            # =========================================
            # KPool rotates q and compressed keys together after pooling.
            # =========================================
            if self.config.dsa_indexer_rotate_activation and self.index_kpool <= 1:
                k = rotate_activation(k)

        # =========================================
        # Prepare weights for index scores.
        # =========================================
        # [seqlen, batch, hidden_size] -> [seqlen, batch, index_n_heads]
        weights = self._project_indexer_weights(x)
        if self.config.dsa_indexer_hf_scoring:
            # HF rounds the projection output to BF16, then upcasts before the
            # head-count scale. The QK scale belongs to the subsequent score GEMM.
            weights = weights.float() * (self.index_n_heads**-0.5)
        else:
            weights = weights * (self.index_n_heads**-0.5) * self.softmax_scale

        # Token-aligned compression scores for the subsequent pool selection.
        kpool_gate_score = None
        if self.index_kpool > 1 and self.index_kpool_compress_gate is not None:
            if self._kpool_trainable:
                # DSAttention calls this method directly, so DSAIndexer's forward pre-hook,
                # which waits for an overlapped parameter all-gather, never runs for these
                # two parameters. A frozen selector keeps them out of DDP: nothing to wait.
                ensure_params_ready([self.index_kpool_compress_gate, self.index_kpool_compress_ape])
            with get_fp8_disabled_context(self.config):
                kpool_gate_score = torch.nn.functional.linear(x, self.index_kpool_compress_gate)
        if return_kpool_gate_score:
            self._kpool_gate_score = None
            return q, k, weights, kpool_gate_score
        self._kpool_gate_score = kpool_gate_score

        return q, k, weights

    def forward_with_scores(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        return_index_scores: bool = True,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass for DSA Indexer that returns both index scores and top-k indices.

        This is used when KL loss is enabled to compare indexer scores with true attention scores.

        Args:
            x: hidden states [seqlen, batch, hidden_size].
            qr: Low-rank query tensor [seqlen, batch, q_lora_rank].
            mask: Optional additive attention mask [seqlen, seqlen] or
                [batch, seqlen, seqlen]. See fused_qk_topk_kpool for KPool support.
            packed_seq_params: Packed sequence parameters for variable length sequences.
            return_index_scores: Skip materializing KPool scores when only top-k is needed.

        Returns:
            index_scores: Index scores [batch, seqlen, seqlen], or None for KPool selection
                when return_index_scores is False.
            topk_indices: Top-k indices [batch, seqlen, index_topk].
        """
        # [seqlen, batch, index_n_heads * index_head_dim]
        # [seqlen, batch, index_head_dim]
        # [seqlen, batch, index_n_heads]
        _reject_mxfp8_indexer_reference_fallback(
            self.config,
            "the DSAIndexer scores and top-k",
            "DSAIndexer.forward_with_scores always scores in BF16",
        )
        q, k, weights = self.forward_before_topk(x, qr, packed_seq_params)

        if self.index_kpool > 1 and self._kpool_gate_score is not None:
            # Select pools, then expand them to token indices.
            _cu_kv = None
            if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
                _, _cu_kv = dsa_layout.get_packed_qk_cu_seqlens(packed_seq_params)
            index_scores, topk_indices = fused_qk_topk_kpool(
                q,
                k,
                weights,
                self.index_topk,
                self.index_kpool,
                self._kpool_gate_score,
                self.index_kpool_compress_ape,
                mask=mask,
                cu_seqlens_kv=_cu_kv,
                use_relu=self.config.dsa_indexer_scoring_relu,
                always_select_tail=self.index_kpool_always_select_tail,
                fp8_indexer=self.index_kpool_use_quantization,
                rotate_activation_enabled=self.config.dsa_indexer_rotate_activation,
                return_index_scores=return_index_scores,
                kpool_key_dtype_compression=self.config.dsa_indexer_kpool_key_dtype_compression,
                hf_scoring=self.config.dsa_indexer_hf_scoring,
            )
            self._kpool_gate_score = None
        else:
            # [batch, seqlen, seqlen], [batch, seqlen, index_topk]
            index_scores, topk_indices = fused_qk_topk_naive(
                q, k, weights, self.index_topk, mask, use_relu=self.config.dsa_indexer_scoring_relu
            )

        return index_scores, topk_indices

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        """
        Forward pass for DSA Indexer.

        Args:
            x: hidden states [seqlen, batch, hidden_size].
            qr: Low-rank query tensor [seqlen, batch, q_lora_rank].
            mask: Attention mask [batch, seqlen, seqlen].
            packed_seq_params: Packed sequence parameters for variable length sequences.

        Returns:
            topk_indices: Top-k indices for sparse attention [batch, seqlen, index_topk].
        """
        _, topk_indices = self.forward_with_scores(
            x, qr, mask, packed_seq_params, return_index_scores=False
        )
        return topk_indices


def unfused_dsa_fn(
    query,
    key,
    value,
    topk_indices,
    softmax_scale,
    mask: Optional[torch.Tensor] = None,
    varlen_starts: Optional[torch.Tensor] = None,
    varlen_ends: Optional[torch.Tensor] = None,
    key_positions: Optional[torch.Tensor] = None,
):
    """
    Unfused sparse attention implementation.

    This path uses chunked sparse softmax accumulation over top-k selected keys
    to avoid materializing full [b, np, sq, skv] attention score tensors.
    """
    if value is None:
        raise NotImplementedError("DSAttention unfused path requires value tensor.")

    query, query_was_thd = dsa_layout.ensure_sbhd(query, "query")
    key, _ = dsa_layout.ensure_sbhd(key, "key")
    value, _ = dsa_layout.ensure_sbhd(value, "value")

    sq, b, np, hn = query.size()
    skv = key.size(0)
    nk = key.size(2)
    hnv = value.size(3)
    nv = value.size(2)

    # [sq, b, np, hn] -> [b, np, sq, hn]
    query_b = query.permute(1, 2, 0, 3).contiguous()
    # [skv, b, nk, hn] -> [b, nk, skv, hn]
    key_b = key.permute(1, 2, 0, 3).contiguous()
    # [skv, b, nv, hnv] -> [b, nv, skv, hnv]
    value_b = value.permute(1, 2, 0, 3).contiguous()
    if nk == 1 and np > 1:
        key_b = key_b.expand(b, np, skv, hn)
    else:
        assert nk == np, "key head count must be 1 (MQA) or match query heads"
    if nv == 1 and np > 1:
        value_b = value_b.expand(b, np, skv, hnv)
    else:
        assert nv == np, "value head count must be 1 (MQA) or match query heads"

    row_mask, varlen_starts, varlen_ends, key_positions = dsa_masking.prepare_sparse_mask_context(
        mask=mask,
        varlen_starts=varlen_starts,
        varlen_ends=varlen_ends,
        key_positions=key_positions,
        sq=sq,
        sk=skv,
        b=b,
        device=query.device,
    )

    seq_chunk_size = 512
    head_chunk_size = 16
    topk_chunk_size = 1024
    safe_k_max = max(0, skv - 1)
    output = torch.empty((sq, b, np * hnv), dtype=value.dtype, device=query.device)

    for bi in range(b):
        for h0 in range(0, np, head_chunk_size):
            h1 = min(h0 + head_chunk_size, np)
            h_chunk = h1 - h0
            out_h0 = h0 * hnv
            out_h1 = h1 * hnv
            k_chunk = key_b[bi, h0:h1, :, :].contiguous()  # [h_chunk, skv, hn]
            v_chunk = value_b[bi, h0:h1, :, :].contiguous()  # [h_chunk, skv, hnv]
            flat_k = k_chunk.reshape(h_chunk * skv, hn)
            flat_v = v_chunk.reshape(h_chunk * skv, hnv)
            head_offsets = (
                torch.arange(h_chunk, device=query.device, dtype=torch.int64).view(-1, 1, 1) * skv
            )

            for s0 in range(0, sq, seq_chunk_size):
                s1 = min(s0 + seq_chunk_size, sq)
                s_len = s1 - s0
                idx_seq_raw = topk_indices[bi, s0:s1]  # [s_len, topk]
                if idx_seq_raw.dtype != torch.int64 or idx_seq_raw.device != query.device:
                    idx_seq_raw = idx_seq_raw.to(dtype=torch.int64, device=query.device)
                valid_seq = idx_seq_raw >= 0
                idx_seq = idx_seq_raw.clamp(min=0, max=safe_k_max)
                q_chunk = query_b[bi, h0:h1, s0:s1, :]  # [h_chunk, s_len, hn]

                # These tensors participate in autograd; reusing cached storage can
                # invalidate saved tensors before backward runs.
                m = torch.full(
                    (h_chunk, s_len), float("-inf"), dtype=torch.float32, device=query.device
                )
                l = torch.zeros((h_chunk, s_len), dtype=torch.float32, device=query.device)
                acc = torch.zeros((h_chunk, s_len, hnv), dtype=torch.float32, device=query.device)

                for t0 in range(0, idx_seq.size(-1), topk_chunk_size):
                    t1 = min(t0 + topk_chunk_size, idx_seq.size(-1))
                    idx_topk = idx_seq[:, t0:t1]  # [s_len, tk]
                    valid_t = valid_seq[:, t0:t1]  # [s_len, tk]
                    flat_idx = idx_topk.unsqueeze(0) + head_offsets  # [h_chunk, s_len, tk]
                    k_sel = flat_k.index_select(0, flat_idx.reshape(-1)).view(
                        h_chunk, s_len, -1, hn
                    )
                    v_sel = flat_v.index_select(0, flat_idx.reshape(-1)).view(
                        h_chunk, s_len, -1, hnv
                    )
                    logits = (q_chunk.float().unsqueeze(2) * k_sel.float()).sum(
                        dim=-1
                    ) * softmax_scale

                    valid_2d, mask_bias = dsa_masking.gather_sparse_topk_validity_and_bias(
                        idx_topk=idx_topk,
                        valid_t=valid_t,
                        bi=bi,
                        s0=s0,
                        s1=s1,
                        row_mask=row_mask,
                        varlen_starts=varlen_starts,
                        varlen_ends=varlen_ends,
                        key_positions=key_positions,
                        dtype=torch.float32,
                    )
                    if mask_bias is not None:
                        logits = logits + mask_bias.unsqueeze(0)
                    logits = logits.masked_fill(
                        ~valid_2d.unsqueeze(0).expand(h_chunk, -1, -1), float("-inf")
                    )
                    m_new = torch.maximum(m, logits.max(dim=-1).values)
                    m_new_for_exp = torch.where(
                        torch.isfinite(m_new), m_new, torch.zeros_like(m_new)
                    )
                    alpha = torch.exp(m - m_new_for_exp)
                    p = torch.exp(logits - m_new_for_exp.unsqueeze(-1))
                    acc = acc * alpha.unsqueeze(-1) + torch.einsum(
                        "hst,hstd->hsd", p, v_sel.float()
                    )
                    l = l * alpha + p.sum(dim=-1)
                    m = m_new

                out_chunk = (acc / l.clamp_min(1e-10).unsqueeze(-1)).to(dtype=value.dtype)
                output[s0:s1, bi, out_h0:out_h1] = out_chunk.permute(1, 0, 2).reshape(
                    s_len, h_chunk * hnv
                )

    if query_was_thd:
        output = output.squeeze(1)
    return output


@dataclass
class _DSAIndexSharingPayload:
    """Ordinary DSA index-sharing tensors carried across physical layers.

    Source/skip layers follow the configured layer-number schedule. Tensors remain
    available to decoder and MTP consumers until the enclosing forward is cleared.

    ``topk_scores_by_layer`` holds the indexer scores aligned slot-for-slot with
    ``topk_by_layer`` and ``topk_key_len_by_layer`` the key length they were selected
    from. Both are written only by the no-grad first call of a checkpointed computing
    layer under ``dsa_indexer_reuse_checkpoint_topk``, together with the indices, so the
    replay can compute the sparse indexer loss without rerunning the dense indexer. With
    ``dsa_indexer_precision='mxfp8'`` the scores are the selected keys' log-probabilities
    over the selection, recomputed in FP32 from the BF16 indexer inputs, which the loss's
    per-row softmax consumes exactly like logits. Any other publication of a layer's
    indices drops its score entries. Without the indexer loss (a frozen indexer) that replay
    stores and needs no scores: it reuses the layer's own ``topk_by_layer``,
    ``topk_length_by_layer`` and ``topk_rows_nonempty_by_layer`` entries, which the
    checkpoint snapshot taken after the first call already holds.

    ``topk_rows_nonempty_by_layer`` holds the publishing call's host-side certificate that
    every row of its top-k selects at least one key (``can_prove_all_topk_rows_nonempty``).
    Every publication writes it, False included, so it never outlives the indices it
    describes; consumers treat a missing entry as False.

    ``frozen_topk_key_len_by_layer`` holds the key length of a frozen indexer's published
    indices. Every publication without the indexer loss writes it next to the indices, and
    the frozen replay checks it before reusing them.
    """

    topk_by_layer: dict[int, torch.Tensor] = field(default_factory=dict)
    topk_length_by_layer: dict[int, torch.Tensor] = field(default_factory=dict)
    topk_scores_by_layer: dict[int, torch.Tensor] = field(default_factory=dict)
    topk_key_len_by_layer: dict[int, int] = field(default_factory=dict)
    topk_rows_nonempty_by_layer: dict[int, bool] = field(default_factory=dict)
    frozen_topk_key_len_by_layer: dict[int, int] = field(default_factory=dict)

    def __copy__(self):
        return type(self)(
            dict(self.topk_by_layer),
            dict(self.topk_length_by_layer),
            dict(self.topk_scores_by_layer),
            dict(self.topk_key_len_by_layer),
            dict(self.topk_rows_nonempty_by_layer),
            dict(self.frozen_topk_key_len_by_layer),
        )


@dataclass
class _DSAFrozenReplayPayload:
    """Layer numbers that one frozen-indexer checkpoint replay has already reached.

    It exists only in the replay's private forward-sharing state. The snapshot keeps one
    top-k entry per layer, written by the layer's last first-call run, so the reuse is valid
    only when the checkpoint runs each layer once. A second replay of the same layer raises,
    because the earlier run's replay would otherwise have used the later run's selection
    without notice.
    """

    layer_numbers: set[int] = field(default_factory=set)

    def __copy__(self):
        return type(self)(set(self.layer_numbers))


@dataclass
class _DSAMTPRepeatedSharingPayload:
    """DSA's repeated-MTP tensors indexed by physical, global layer number.

    Separate from ordinary IndexShare. Source calls replace only their layer's
    entries; consumers use MTP's explicit source/consumer flag. The enclosing
    forward owns tensor cleanup, while snapshots retain independent dictionaries.
    ``topk_rows_nonempty_by_layer`` is the source's non-empty top-k row certificate,
    published and cleared together with its indices; a missing entry means False.
    """

    topk_by_layer: dict[int, torch.Tensor] = field(default_factory=dict)
    topk_length_by_layer: dict[int, torch.Tensor] = field(default_factory=dict)
    latent_kv_by_layer: dict[int, torch.Tensor] = field(default_factory=dict)
    topk_rows_nonempty_by_layer: dict[int, bool] = field(default_factory=dict)

    def __copy__(self):
        return type(self)(
            dict(self.topk_by_layer),
            dict(self.topk_length_by_layer),
            dict(self.latent_kv_by_layer),
            dict(self.topk_rows_nonempty_by_layer),
        )


class DSAttention(MegatronModule):
    """
    This module implements sparse attention mechanism using an DSA Indexer to compute top-k
    attention indices for reducing computational complexity.

    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L491-L597
    """

    consumes_absorbed_v_up_projection = True
    requires_dsa_inputs = True
    logs_dsa_indexer_loss = True
    _LAYOUT_HOLDER_ATTR = dsa_layout.PACKED_CP_LAYOUT_MEMO_ATTR

    def __init__(
        self,
        config: TransformerConfig,
        submodules: DSAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        k_channels: Optional[int] = None,
        v_channels: Optional[int] = None,
        cp_comm_type: str = "p2p",
        pg_collection: ProcessGroupCollection = None,
        is_mtp_layer: bool = False,
    ):
        super().__init__(config=config)

        self.layer_number = layer_number + self.config.num_layers if is_mtp_layer else layer_number
        self.index_topk = self.config.dsa_indexer_topk
        self.index_topk_freq = self.config.dsa_indexer_topk_freq or 1
        self.index_skip_topk_offset = self.config.dsa_indexer_skip_topk_offset or 0
        self.index_share = self.index_topk_freq > 1
        self.skip_topk = self.index_share and is_dsa_skip_topk_layer(
            self.layer_number, self.index_skip_topk_offset, self.index_topk_freq
        )
        mtp_shared_components = frozenset(self.config.mtp_repeated_layer_shared_components or ())
        self.mtp_latent_kv_share = "latent_kv" in mtp_shared_components and is_mtp_layer
        self.mtp_sparse_attention_index_share = (
            "sparse_attention_index" in mtp_shared_components and is_mtp_layer
        )
        self.mtp_cross_depth_share = (
            self.mtp_latent_kv_share or self.mtp_sparse_attention_index_share
        )
        global _warned_mtp_index_sharing_fused_bypass
        if (
            self.mtp_sparse_attention_index_share
            and dsa_kernels.use_fused_dsa_kernels(self.config)
            and not _warned_mtp_index_sharing_fused_bypass
        ):
            _warned_mtp_index_sharing_fused_bypass = True
            log_single_rank(
                logger,
                logging.WARNING,
                "MTP sparse_attention_index sharing requires reusable top-k indices, which the "
                "combined fused DSA kernel does not expose. The repeated MTP layer therefore "
                "uses the separable top-k and sparse-attention path; configured fused kernels "
                "for those operations remain eligible. Benchmark this tradeoff for the target "
                "MTP depth.",
            )
        self.source_layer = (
            source_dsa_compute_layer(
                self.layer_number, self.index_skip_topk_offset, self.index_topk_freq
            )
            if self.index_share
            else self.layer_number
        )

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        self.pg_collection = pg_collection

        self.indexer = None
        if not self.skip_topk:
            self.indexer = build_module(
                submodules.indexer, config=self.config, pg_collection=self.pg_collection
            )

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(
                k_channels if k_channels is not None else config.kv_channels
            )
        self.softmax_scale = softmax_scale
        self.cp_comm_type = dsa_layout.normalize_cp_comm_type(cp_comm_type)

    def _get_forward_sharing_state(
        self, packed_seq_params: Optional[PackedSeqParams], attention_mask: Optional[torch.Tensor]
    ):
        """Return the generic state attached to this forward's canonical carrier."""
        return get_forward_sharing_state(
            packed_seq_params=packed_seq_params, attention_mask=attention_mask, config=self.config
        )

    def _get_dsa_index_sharing_payload(
        self, packed_seq_params: Optional[PackedSeqParams], attention_mask: Optional[torch.Tensor]
    ) -> _DSAIndexSharingPayload:
        """Return DSA's global payload for ordinary cross-layer index sharing."""
        return self._get_forward_sharing_state(packed_seq_params, attention_mask).get_or_create(
            _DSAIndexSharingPayload
        )

    def _get_packed_cp_layout_cache(
        self, packed_seq_params: Optional[PackedSeqParams]
    ) -> Optional[dict]:
        """Return the per-microbatch memo for packed-CP layout metadata, or None.

        The CP query positions and key reorder indices depend only on
        ``cu_seqlens_q``/``cu_seqlens_kv``, ``cp_size``, ``cp_rank`` and the
        requested output sizes. All of those are identical for every layer in a
        microbatch, yet each layer rebuilds them, and each rebuild loops
        ``cp_size`` times over :func:`build_packed_allgather_cp_local_positions`,
        whose boolean-mask indexing has data-dependent output shapes and so
        forces a device-to-host size readback.

        ``PackedSeqParams`` is the only carrier used here. It is constructed per
        microbatch, so a memo hung on it cannot outlive the layout it describes
        -- which matters under dynamic CP, where the layout changes between
        microbatches. The index-share top-k holders fall back to
        ``attention_mask``/``self.config``, but that is only safe because every
        computing layer overwrites its slot before any sharing layer reads it. A
        layout memo has no such write-before-read ordering and ``self.config``
        outlives the microbatch, so caching there could serve a stale layout.
        """
        if packed_seq_params is None:
            return None
        cache = getattr(packed_seq_params, self._LAYOUT_HOLDER_ATTR, None)
        if cache is None:
            cache = {}
            setattr(packed_seq_params, self._LAYOUT_HOLDER_ATTR, cache)
        return cache

    def _kpool_pool_topk_fn(
        self,
        *,
        packed_thd: bool,
        has_mask: bool,
        positions_from_position_ids: bool,
        local_rows: int,
        full_rows: int,
        row_start: int,
        skv: int,
        cp_size: int,
        cp_rank: int,
        pool_scores_out: Optional[dict] = None,
    ):
        """The fused pool selector for ``fused_qk_topk_kpool``, or None for the PyTorch scorer.

        With ``dsa_indexer_kpool_scorer_backend='cudnn'``, returns a callable that hands the
        pooled keys to the DSA kernel backend's ``run_kpool_pool_topk`` with this call's query
        positions as host-side runs. They are the positions ``build_dsattention_forward_mask``
        builds the causal bounds from, so both scorers apply the same causal rule. Layouts
        whose positions are not host facts (packed THD, explicit masks, query positions taken
        from ``position_ids``, non-uniform CP) keep the PyTorch scorer, warned once per reason.
        ``positions_from_position_ids`` says that the forward reads the query positions from
        ``position_ids``, which it does only under context parallelism or for a TP
        sequence-parallel local query; ``position_ids`` passed without either are not read
        and do not keep the PyTorch scorer.

        With ``pool_scores_out`` (a dict), the selector also asks the backend for the score of
        each selected pool and stores it under ``"pool_scores"`` (``[batch, sq, pool_topk]``
        FP32, aligned with the pool ids it returns), the student of the cuDNN KPool loss;
        the key stays unset when the scorer declines. It still returns only the pool ids, so
        ``fused_qk_topk_kpool`` and the selection are the same as without it.
        """
        if self.config.dsa_indexer_kpool_scorer_backend != "cudnn":
            return None
        query_runs = None
        if not dsa_kernels.use_fused_dsa_kernels(self.config):
            reason = "the fused DSA kernels are disabled (attention_backend is unfused)"
        elif packed_thd:
            reason = "packed THD sequences restart pools at every document"
        elif has_mask:
            reason = "an explicit attention mask replaces the causal layout"
        elif positions_from_position_ids:
            reason = "the query positions come from position_ids"
        else:
            query_runs, reason = dsa_layout.nonpacked_query_position_runs(
                local_rows=local_rows,
                full_rows=full_rows,
                row_start=row_start,
                skv=skv,
                cp_size=cp_size,
                cp_rank=cp_rank,
            )
        if query_runs is None:
            _warn_kpool_scorer_fallback_once(reason)
            return None
        config = self.config

        def select_pools(q, k_pooled, weights, pool_topk, pool_size, hf_scoring):
            if pool_scores_out is None:
                return dsa_kernels.run_kpool_pool_topk(
                    config,
                    q,
                    k_pooled,
                    weights,
                    pool_topk,
                    pool_size,
                    query_runs,
                    hf_scoring,
                    use_relu=config.dsa_indexer_scoring_relu,
                )
            result = dsa_kernels.run_kpool_pool_topk(
                config,
                q,
                k_pooled,
                weights,
                pool_topk,
                pool_size,
                query_runs,
                hf_scoring,
                use_relu=config.dsa_indexer_scoring_relu,
                return_scores=True,
            )
            if result is None:
                return None
            pool_ids, pool_scores = result
            if pool_scores is not None:
                pool_scores_out["pool_scores"] = pool_scores
            return pool_ids

        return select_pools

    def _kpool_pool_then_gather(
        self,
        k: torch.Tensor,
        gate_score: Optional[torch.Tensor],
        *,
        cu_seqlens_kv: Optional[torch.Tensor],
        cp_size: int,
        cp_group: Optional[torch.distributed.ProcessGroup],
        local_rows: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pool this rank's indexer keys, then all-gather the pooled keys over CP.

        The trainable KPool path compresses the rank's own tokens under the caller's grad
        mode and gathers ``[local_pools, batch, dim]`` pooled keys, not the token-level
        indexer K and gate. The gather runs in FP32, which keeps the values exact and makes
        its backward reduce-scatter sum the CP ranks' pooled-key gradients in FP32; the
        result is cast back to the pooled dtype. Each zigzag half holds whole pools
        (DSAttention checks ``local_rows % (2 * pool_size) == 0``), so pooling commutes with
        the gather, and the zigzag reorder built at pool granularity puts the pools in
        sequence order: bit for bit the pools that the frozen path compresses from the
        gathered keys. Packed THD (without CP) pools per document from ``cu_seqlens_kv``.

        Returns:
            ``(k_pooled [pools, batch, dim], pool_token_base [pools])`` in global order.
        """
        k_pooled, pool_token_base = self.indexer.kpool_pool_keys(
            k, gate_score, cu_seqlens=cu_seqlens_kv
        )
        if cp_size <= 1:
            return k_pooled, pool_token_base
        pool_size = self.indexer.index_kpool
        local_pools = k_pooled.size(0)
        if k.size(0) != local_rows or local_pools * pool_size != local_rows:
            raise RuntimeError(
                "DSA trainable KPool expects whole pools of this CP rank's indexer keys: "
                f"key_rows={k.size(0)}, local_rows={local_rows}, pools={local_pools}, "
                f"pool_size={pool_size}."
            )
        pool_reorder_idx, pool_reorder_inv_idx = dsa_layout.build_zigzag_allgather_cp_key_reorder(
            sq=local_pools, cp_size=cp_size, device=k.device, return_inverse_key_reorder=True
        )
        pooled_dtype = k_pooled.dtype
        gathered = gather_from_sequence_parallel_region(k_pooled.float(), group=cp_group)
        if gathered.size(0) != pool_reorder_idx.numel():
            raise RuntimeError(
                "DSA gathered pooled-key length mismatch: "
                f"pools={gathered.size(0)}, expected={pool_reorder_idx.numel()}"
            )
        k_pooled = _permute_rows(gathered, pool_reorder_idx, pool_reorder_inv_idx).to(pooled_dtype)
        pool_token_base = (
            torch.arange(k_pooled.size(0), device=k_pooled.device, dtype=torch.int64) * pool_size
        )
        return k_pooled, pool_token_base

    @staticmethod
    def _broadcast_kpool_selection_from_tp_rank0(
        topk_indices: torch.Tensor, pool_ids: torch.Tensor, tp_group: torch.distributed.ProcessGroup
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Give every TP rank the token top-k and pool ids that TP rank 0 selected."""
        src = torch.distributed.get_process_group_ranks(tp_group)[0]
        topk_indices = topk_indices.contiguous()
        pool_ids = pool_ids.contiguous()
        torch.distributed.broadcast(topk_indices, src=src, group=tp_group)
        torch.distributed.broadcast(pool_ids, src=src, group=tp_group)
        return topk_indices, pool_ids

    def _kpool_sparse_indexer_loss(
        self,
        *,
        q: torch.Tensor,
        weights: torch.Tensor,
        k_pooled: torch.Tensor,
        topk_indices: torch.Tensor,
        pool_ids: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        query_valid_rows: Optional[torch.Tensor],
        indexer_loss_coeff: float,
        tp_group: Optional[torch.distributed.ProcessGroup],
        metric_num_layers: int,
        metric_reduce_group: Optional[torch.distributed.ProcessGroup],
        metric_avg_group: Optional[torch.distributed.ProcessGroup],
    ) -> torch.Tensor:
        """The sparse pool-level KL of a trainable KPool layer, attached to its indexer graph.

        Canonicalizes the selected slots, builds the teacher from the detached main query and
        key over every attended token (pools plus tail), and applies
        ``KPoolSparseIndexerLoss`` to the indexer's own scoring inputs, rotated as the
        selector rotates them. The rows and their reduction are the token loss's:
        ``query_valid_rows`` (or every row) divides the summed KL, or nothing does under
        ``calculate_per_token_loss``; rows without pool mass add zero.

        It also records this layer's tail mass ``p_E`` in the metric tracker, apart from the
        indexer loss: the loss drops the per-row ``1 - p_E`` weight, and the logged ``p_E``
        shows what that leaves out. The tracker gets its sum over the rows the loss uses
        (``row_has_mass``, and ``query_valid_rows`` when given) and their number.
        ``metric_*`` are the indexer loss's tracker arguments.
        """
        pool_size = self.indexer.index_kpool
        pool_ids_c, topk_indices_c = _kpool_canonical_slots(
            topk_indices, pool_ids, index_topk=self.index_topk, pool_size=pool_size
        )
        target, row_has_mass, tail_mass = _kpool_sparse_pool_target(
            query.detach(),
            key.detach(),
            topk_indices_c,
            pool_ids_c,
            softmax_scale=self.softmax_scale,
            index_topk=self.index_topk,
            pool_size=pool_size,
            tp_group=tp_group,
        )
        row_mask, row_count = _kpool_loss_rows(
            row_has_mass,
            tail_mass,
            query_valid_rows=query_valid_rows,
            calculate_per_token_loss=self.config.calculate_per_token_loss,
            layer_number=self.layer_number,
            num_layers=metric_num_layers,
            reduce_group=metric_reduce_group,
            avg_group=metric_avg_group,
        )
        # The selector scores q and the pooled keys after a Hadamard rotation, when enabled.
        q, k_pooled = _kpool_scoring_inputs(q, k_pooled, self.config.dsa_indexer_rotate_activation)
        return KPoolSparseIndexerLoss.apply(
            q,
            weights,
            k_pooled,
            pool_ids_c,
            target,
            row_mask,
            row_count,
            indexer_loss_coeff,
            self.config.dsa_indexer_hf_scoring,
            self.config.dsa_indexer_scoring_relu,
            _KPOOL_LOSS_CHUNK_BYTES,
        )

    def _kpool_cudnn_loss_decline_reason(
        self, *, tp_size: int, absorbed_mla: bool, query: torch.Tensor, key: torch.Tensor
    ) -> Optional[str]:
        """Why this forward's trainable KPool loss cannot run on the cuDNN frontend kernels.

        None when it can, as far as host facts decide before selection; the cuDNN KPool
        scorer must then still accept the forward and return its pool scores. The cuDNN path
        needs TP size 1 (TP ranks use TP rank 0's selection, which another rank's top-k values
        do not belong to), no deterministic mode, ``dsa_kernel_backend='cudnn'``, the cuDNN
        KPool scorer (its top-k values are the student), HF scoring, the fused absorbed-MLA
        sparse attention (its LSE feeds the teacher) and cuDNN frontend >= 1.31 (the backend
        checks the last two).

        Deterministic mode keeps the PyTorch loss: its pooled-key gradient is free of atomics
        at a cost that grows with the selected pools, whereas the cuDNN path's atomic-free
        replacement (``_deterministic_sparse_indexer_grads_wk``) sweeps every pooled key in
        each query-row chunk, and its fixed per-key buffers shrink the chunks to one row at
        1M tokens (262,144 pooled keys).
        """
        config = self.config
        if tp_size > 1:
            return (
                f"tensor parallelism (TP size {tp_size}): every TP rank attends TP rank 0's "
                "selection, which another rank's cuDNN top-k values do not belong to"
            )
        if dsa_kernels._deterministic_dsa_requested(config):
            return (
                "deterministic mode: the cuDNN path's atomic-free pooled-key gradient sweeps "
                "every pooled key in each query-row chunk (one row per chunk at 1M tokens), "
                "while the PyTorch loss accumulates it over the selected pools only"
            )
        if config.dsa_kernel_backend != "cudnn":
            return f"dsa_kernel_backend is {config.dsa_kernel_backend!r}, not 'cudnn'"
        if config.dsa_indexer_kpool_scorer_backend != "cudnn":
            return (
                "dsa_indexer_kpool_scorer_backend is "
                f"{config.dsa_indexer_kpool_scorer_backend!r}, not 'cudnn'"
            )
        if not config.dsa_indexer_hf_scoring:
            return "the indexer uses legacy (non-HF) KPool scoring"
        if not dsa_kernels.use_fused_dsa_kernels(config):
            return "the fused DSA kernels are disabled (attention_backend is unfused)"
        if not absorbed_mla:
            return (
                "the main attention is not absorbed MLA, so no fused sparse attention returns "
                "the LSE that the teacher reads"
            )
        return dsa_kernels.kpool_sparse_indexer_loss_decline_reason(config, query, key)

    def _kpool_sparse_indexer_loss_cudnn(
        self,
        *,
        q: torch.Tensor,
        weights: torch.Tensor,
        k_pooled: torch.Tensor,
        topk_indices: torch.Tensor,
        pool_ids: torch.Tensor,
        pool_scores: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        lse: torch.Tensor,
        query_valid_rows: Optional[torch.Tensor],
        indexer_loss_coeff: float,
        metric_num_layers: int,
        metric_reduce_group: Optional[torch.distributed.ProcessGroup],
        metric_avg_group: Optional[torch.distributed.ProcessGroup],
    ) -> torch.Tensor:
        """``_kpool_sparse_indexer_loss`` on cuDNN frontend kernels, after the sparse attention.

        The teacher comes from FE score recompute on the selector's own token ids
        (``topk_indices``: pool slots in selector order, then the tail), fed the LSE that the
        fused sparse attention returned for them; each pool's slots are summed, and the
        normalization, rows, divisor and ``p_E`` record are phase 1's. The student is the
        cuDNN selector's own score of each selected pool (``pool_scores``, aligned with
        ``pool_ids``), and the gradients come from the FE sparse indexer backward on the
        pooled keys. TP size 1 only (see ``_kpool_cudnn_loss_decline_reason``), so slots
        need no canonical order and the target no TP reduction.
        """
        token_target = dsa_kernels.run_kpool_attention_token_target(
            self.config,
            query=query.detach(),
            key=key.detach(),
            lse=lse,
            token_topk=topk_indices,
            softmax_scale=self.softmax_scale,
        )
        target, row_has_mass, tail_mass = _kpool_pool_target_from_token_target(
            token_target, pool_ids, index_topk=self.index_topk, pool_size=self.indexer.index_kpool
        )
        del token_target
        row_mask, row_count = _kpool_loss_rows(
            row_has_mass,
            tail_mass,
            query_valid_rows=query_valid_rows,
            calculate_per_token_loss=self.config.calculate_per_token_loss,
            layer_number=self.layer_number,
            num_layers=metric_num_layers,
            reduce_group=metric_reduce_group,
            avg_group=metric_avg_group,
        )
        q, k_pooled = _kpool_scoring_inputs(q, k_pooled, self.config.dsa_indexer_rotate_activation)
        return dsa_kernels.run_kpool_sparse_indexer_loss(
            self.config,
            q=q,
            weights=weights,
            k_pooled=k_pooled,
            pool_ids=pool_ids,
            pool_scores=pool_scores,
            target=target,
            row_mask=row_mask,
            row_count=row_count,
            loss_coeff=indexer_loss_coeff,
            hf_scoring=self.config.dsa_indexer_hf_scoring,
            use_relu=self.config.dsa_indexer_scoring_relu,
        )

    @staticmethod
    def _memoized(cache: Optional[dict], key: tuple, build, prebuilt=None):
        """Return ``cache[key]``, building it on first use. No-op when cache is None.

        ``prebuilt`` (tables built at batch construction for exactly this request) is
        returned as-is instead, without building or touching the cache.
        """
        if prebuilt is not None:
            return prebuilt
        if cache is None:
            return build()
        if key not in cache:
            cache[key] = build()
        return cache[key]

    @staticmethod
    def _match_prebuilt_packed_cp_layout(
        packed_seq_params: Optional[PackedSeqParams],
        requested: dsa_layout.PackedCPLayoutGeometry,
        device: torch.device,
    ) -> Optional[dsa_layout.PrebuiltPackedCPLayout]:
        """Return the batch-time prebuilt packed-CP layout if it answers ``requested``.

        ``None`` selects the lazy memo path: when no current layout is attached (see
        ``dsa_layout.get_prebuilt_packed_cp_layout``), or when the attached one describes
        another geometry or device (warned once per kind of mismatch), since the lazy
        path builds the requested layout from the same host boundaries. Inside a CUDA
        graph capture both raise instead: a lazily built table, or a memo entry from an
        eager warmup, would be baked into the graph and go stale when the packing
        changes. A layout whose tables contradict its own geometry raises (see
        ``PrebuiltPackedCPLayout.validate``).
        """
        layout = dsa_layout.get_prebuilt_packed_cp_layout(packed_seq_params)
        if layout is not None and layout.geometry == requested and layout.device == device:
            layout.validate()
            return layout
        capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        if layout is None:
            if capturing:
                raise RuntimeError(
                    "DSA cannot build its packed-CP layout inside CUDA graph capture: no current "
                    f"prebuilt layout for {requested} is attached to the PackedSeqParams it "
                    "received. Prebuild it outside the capture with "
                    "dsa_layout.prebuild_packed_cp_layout (prepare_packed_seq_params does this "
                    "for DSA models) and keep it attached to the PackedSeqParams DSA consumes."
                )
            return None
        mismatch = (
            f"the prebuilt packed-CP layout was built for {layout.geometry} on {layout.device}, "
            f"but this DSA layer requests {requested} on {device}"
        )
        if capturing:
            raise RuntimeError(
                f"DSA cannot use a lazily built packed-CP layout inside CUDA graph capture: "
                f"{mismatch}. Refresh the static prebuilt layout for this geometry before "
                "capture/replay."
            )
        kind = tuple(
            name
            for name in requested._fields
            if getattr(layout.geometry, name) != getattr(requested, name)
        ) + (("device",) if layout.device != device else ())
        if kind not in _DSA_PREBUILT_LAYOUT_MISMATCH_WARNED:
            _DSA_PREBUILT_LAYOUT_MISMATCH_WARNED.add(kind)
            logger.warning(
                "DSA ignores a prebuilt layout and builds it in the forward instead, so every "
                "microbatch with this mismatch builds the tables twice: %s.",
                mismatch,
            )
        return None

    def _prepare_mtp_sharing_payload(
        self, packed_seq_params: Optional[PackedSeqParams], attention_mask: Optional[torch.Tensor]
    ) -> Tuple[Optional[bool], Optional[_DSAMTPRepeatedSharingPayload]]:
        """Clear this source layer's old tensors or require its published tensors."""
        if not self.mtp_cross_depth_share:
            return None, None
        state = self._get_forward_sharing_state(packed_seq_params, attention_mask)
        is_source = is_mtp_repeated_sharing_source(state, self.layer_number)
        if is_source:
            payload = state.get_or_create(_DSAMTPRepeatedSharingPayload)
            payload.topk_by_layer.pop(self.layer_number, None)
            payload.topk_length_by_layer.pop(self.layer_number, None)
            payload.topk_rows_nonempty_by_layer.pop(self.layer_number, None)
            payload.latent_kv_by_layer.pop(self.layer_number, None)
        else:
            payload = self._require_mtp_sharing_payload(packed_seq_params, attention_mask)
        return is_source, payload

    def _require_mtp_sharing_payload(
        self, packed_seq_params: Optional[PackedSeqParams], attention_mask: Optional[torch.Tensor]
    ) -> _DSAMTPRepeatedSharingPayload:
        """Require the tensors published by this physical layer's source call."""
        state = self._get_forward_sharing_state(packed_seq_params, attention_mask)
        is_mtp_repeated_sharing_source(state, self.layer_number)
        payload = state.get(_DSAMTPRepeatedSharingPayload)
        if payload is None:
            raise RuntimeError("MTP sharing consumer requires published source tensors.")
        if self.mtp_latent_kv_share and self.layer_number not in payload.latent_kv_by_layer:
            raise RuntimeError("MTP sharing consumer requires published source latent KV.")
        if self.mtp_sparse_attention_index_share and self.layer_number not in payload.topk_by_layer:
            raise RuntimeError("MTP sharing consumer requires published source indices.")
        return payload

    def get_mtp_checkpoint_tensors(
        self, packed_seq_params: PackedSeqParams | None, attention_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, ...]:
        """Export this layer's shared tensors as explicit checkpoint outputs/inputs.

        The tuple layout is private to DSA. Empty top-k lengths encode an absent
        optional tensor, since checkpoint outputs must be tensors.
        """
        payload = self._require_mtp_sharing_payload(packed_seq_params, attention_mask)
        tensors = []
        if self.mtp_latent_kv_share:
            tensors.append(payload.latent_kv_by_layer[self.layer_number])
        if self.mtp_sparse_attention_index_share:
            indices = payload.topk_by_layer[self.layer_number]
            lengths = payload.topk_length_by_layer.get(self.layer_number)
            tensors.extend((indices, lengths if lengths is not None else indices.new_empty(0)))
        return tuple(tensors)

    def set_mtp_checkpoint_tensors(
        self,
        tensors: tuple[torch.Tensor, ...],
        packed_seq_params: PackedSeqParams | None,
        attention_mask: torch.Tensor | None,
    ) -> None:
        """Bind graph-connected checkpoint outputs or detached replay inputs.

        The host-side top-k row certificate is not a tensor and stays in the payload. It
        remains valid because every binding passes the values that this payload's own
        source call published: the source's checkpoint outputs, a consumer's inputs taken
        from them, or those inputs replayed against the snapshot that saved them.
        """
        state = self._get_forward_sharing_state(packed_seq_params, attention_mask)
        payload = state.get_or_create(_DSAMTPRepeatedSharingPayload)
        expected = int(self.mtp_latent_kv_share) + 2 * int(self.mtp_sparse_attention_index_share)
        if len(tensors) != expected:
            raise ValueError(
                f"Expected {expected} repeated-MTP checkpoint tensors, got {len(tensors)}"
            )
        offset = 0
        if self.mtp_latent_kv_share:
            payload.latent_kv_by_layer[self.layer_number] = tensors[0]
            offset = 1
        if self.mtp_sparse_attention_index_share:
            indices, lengths = tensors[offset:]
            payload.topk_by_layer[self.layer_number] = indices
            if lengths.numel():
                payload.topk_length_by_layer[self.layer_number] = lengths
            else:
                payload.topk_length_by_layer.pop(self.layer_number, None)

    def should_reuse_mtp_latent_kv(
        self,
        packed_seq_params: Optional[PackedSeqParams],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> bool:
        """Consult the MTP-published role before computing absorbed MLA projections."""
        if not self.mtp_latent_kv_share:
            return False
        state = self._get_forward_sharing_state(packed_seq_params, attention_mask)
        return not is_mtp_repeated_sharing_source(state, self.layer_number)

    def backward_dw(self):
        """Compute the deferred weight gradients (delay_wgrad_compute) of the indexer."""
        if self.indexer is not None:
            self.indexer.backward_dw()

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        x: torch.Tensor,
        qr: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attn_mask_type: AttnMaskType = None,
        attention_bias: torch.Tensor = None,
        packed_seq_params: PackedSeqParams = None,
        up_v_weight: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for Sparse Attention.

        Args:
            query: Query tensor [sq, b, np, hn] or packed [t, np, hn].
            key: Key tensor [skv, b, np, hn] or packed [t, np, hn]. May be None only for a
                later repeated-MTP depth that reuses latent KV from the DSA sharing state.
            value: Value tensor [skv, b, np, hnv] or packed [t, np, hnv].
            x: Original hidden states [sq, b, hidden_size].
            qr: Low-rank query representation [sq, b, q_lora_rank].
            position_ids: Optional position ids [b, sq], used by allgather CP causal masking.
            attention_mask: Attention mask tensor [b, 1, sq, sk].
            attn_mask_type: Type of attention mask.
            attention_bias: Optional attention bias.
            packed_seq_params: Packed sequence parameters.

        Returns:
            output: Output tensor [sq, b, hidden_size]
        """
        is_mtp_source, mtp_payload = self._prepare_mtp_sharing_payload(
            packed_seq_params, attention_mask
        )
        reuse_sparse_attention_index = bool(
            self.mtp_sparse_attention_index_share and mtp_payload is not None and not is_mtp_source
        )
        reuse_latent_kv = bool(
            self.mtp_latent_kv_share and mtp_payload is not None and not is_mtp_source
        )
        if reuse_latent_kv:
            if key is not None:
                raise RuntimeError(
                    "A repeated-MTP latent-KV consumer must obtain key from the DSA sharing "
                    "state instead of receiving a newly computed key."
                )
            assert mtp_payload is not None
            key = mtp_payload.latent_kv_by_layer[self.layer_number]
        elif key is None:
            raise RuntimeError("DSAttention requires a key tensor outside latent-KV reuse.")
        query, _ = dsa_layout.ensure_sbhd(query, "query")
        key, _ = dsa_layout.ensure_sbhd(key, "key")
        if value is not None:
            value, _ = dsa_layout.ensure_sbhd(value, "value")
        if up_v_weight is not None:
            assert up_v_weight.ndim == 3, "up_v_weight must be [heads, v_head_dim, kv_lora_rank]"
            up_v_weight = up_v_weight.to(device=query.device, dtype=query.dtype).contiguous()
            if value is not None:
                raise RuntimeError(
                    "DSAttention received up_v_weight with explicit value tensor. "
                    "For absorbed DSA path, value must be None."
                )

        latent_v_channels = int(getattr(self.config, "kv_lora_rank", 0) or 0)
        qk_pos_dim = int(getattr(self.config, "qk_pos_emb_head_dim", 0) or 0)
        expected_absorbed_dim = latent_v_channels + qk_pos_dim
        absorbed_mla = (
            latent_v_channels > 0
            and expected_absorbed_dim > 0
            and key.size(2) == 1
            and query.size(-1) == key.size(-1) == expected_absorbed_dim
        )
        if value is None and not absorbed_mla:
            raise RuntimeError(
                "DSAttention received value=None but query/key are not in absorbed layout. "
                f"query_hdim={query.size(-1)}, key_hdim={key.size(-1)}, key_heads={key.size(2)}, "
                f"expected_absorbed_dim={expected_absorbed_dim}"
            )
        if up_v_weight is not None and not absorbed_mla:
            raise RuntimeError(
                "DSAttention received up_v_weight but absorbed layout was not detected. "
                f"query_hdim={query.size(-1)}, key_hdim={key.size(-1)}, key_heads={key.size(2)}, "
                f"expected_absorbed_dim={expected_absorbed_dim}"
            )

        sq, b, _, _ = query.size()
        local_sequence_rows = x.size(0)

        cp_group = getattr(self.pg_collection, "cp", None)
        cp_size = get_pg_size(cp_group)
        cp_rank = cp_group.rank() if cp_group is not None else 0
        tp_group = getattr(self.pg_collection, "tp", None)
        tp_size = get_pg_size(tp_group)
        sequence_parallel_tp = self.config.sequence_parallel and tp_size > 1
        sequence_parallel_tp_row_start = 0
        sequence_parallel_tp_full_rows = sq
        sequence_parallel_query_is_local = False
        if sequence_parallel_tp:
            sequence_parallel_tp_full_rows = local_sequence_rows * tp_size
            if sq == local_sequence_rows:
                sequence_parallel_query_is_local = True
                sequence_parallel_tp_row_start = tp_group.rank() * local_sequence_rows
            elif sq != sequence_parallel_tp_full_rows:
                raise RuntimeError(
                    "DSA sequence-parallel query row count mismatch: "
                    f"query_rows={sq}, local_rows={local_sequence_rows}, tp_size={tp_size}"
                )
        cp_local_sequence_rows = sequence_parallel_tp_full_rows
        cp_global_sequence_rows = cp_local_sequence_rows * cp_size
        packed_thd = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
        packed_query_positions = None
        nonpacked_query_positions = None
        kv_reorder_idx = None
        # Inverse of kv_reorder_idx, built (for packed THD, memoized) with it; it gives the
        # reorder a gather backward.
        kv_reorder_inv_idx = None
        single_packed_thd_sequence = False
        # Layout metadata is identical for every layer in a microbatch; memoize it on
        # the per-microbatch PackedSeqParams so only the first DSA layer pays for it.
        layout_cache = self._get_packed_cp_layout_cache(packed_seq_params)
        if packed_thd:
            cu_seqlens_q, cu_seqlens_kv = dsa_layout.get_packed_qk_cu_seqlens(packed_seq_params)
            # Host copies of the compacted cu_seqlens, stored by
            # prebuild_thd_cp_partition_routes at batch-construction time. When
            # present, the layout builders below run entirely on the host: no
            # kernels, no device readbacks, one async copy of the finished table.
            host_cu_q = getattr(packed_seq_params, "thd_cp_host_cu_seqlens_q", None)
            host_cu_kv = getattr(packed_seq_params, "thd_cp_host_cu_seqlens_kv", None)
            # Whether the pack holds one sequence is a fact about the pack, not about
            # context parallelism; which kernels accept that layout is the scoring
            # plan's decision. Consumers that genuinely need CP already test cp_size.
            single_packed_thd_sequence = cu_seqlens_q.numel() == 2 and cu_seqlens_kv.numel() == 2
            packed_query_output_size = (
                sequence_parallel_tp_full_rows if sequence_parallel_tp else sq
            )
            packed_global_output_size = packed_query_output_size * cp_size
            query_cu_seqlens_cover_output, key_cu_seqlens_cover_output = (
                dsa_layout.resolve_packed_cp_cover_output_flags(
                    packed_seq_params, packed_global_output_size
                )
            )

            def _prebuilt_layout_tables(local_rows, positions_only=False):
                # Tables prebuilt at batch construction (dsa_layout.prebuild_packed_cp_layout)
                # for exactly this request, shaped like the memo entry they replace; None
                # selects the memo path below (and raises inside a CUDA graph capture).
                layout = self._match_prebuilt_packed_cp_layout(
                    packed_seq_params,
                    dsa_layout.PackedCPLayoutGeometry(
                        cp_size=cp_size,
                        cp_rank=cp_rank,
                        local_rows=local_rows,
                        query_cu_seqlens_cover_output=query_cu_seqlens_cover_output,
                        key_cu_seqlens_cover_output=key_cu_seqlens_cover_output,
                    ),
                    query.device,
                )
                if layout is None:
                    return None
                # The SP "local_positions" entry comes from the same host positions
                # builder with the same output size as the reorder wrapper's positions.
                if positions_only:
                    return layout.query_positions
                return layout.query_positions, layout.kv_reorder_idx, layout.kv_reorder_inv_idx

            if sequence_parallel_query_is_local and cp_size == 1:
                row_start = sequence_parallel_tp_row_start
                packed_query_positions = torch.arange(
                    row_start, row_start + sq, dtype=torch.int64, device=query.device
                )
            elif sequence_parallel_tp and cp_size > 1:
                packed_query_positions_full = self._memoized(
                    layout_cache,
                    ("local_positions", cp_size, cp_rank, packed_query_output_size),
                    lambda: (
                        _build_cp_positions_from_host(
                            host_cu_q,
                            cp_size,
                            cp_rank,
                            query.device,
                            output_size=packed_query_output_size,
                        )
                        if host_cu_q is not None
                        else dsa_layout.build_packed_allgather_cp_local_positions(
                            cu_seqlens_q,
                            cp_size,
                            cp_rank,
                            query.device,
                            output_size=packed_query_output_size,
                        )
                    ),
                    prebuilt=_prebuilt_layout_tables(packed_query_output_size, positions_only=True),
                )
                if sequence_parallel_query_is_local:
                    row_start = sequence_parallel_tp_row_start
                    packed_query_positions = packed_query_positions_full[row_start : row_start + sq]
                else:
                    packed_query_positions = packed_query_positions_full
            elif cp_size > 1:
                packed_query_positions, kv_reorder_idx, kv_reorder_inv_idx = self._memoized(
                    layout_cache,
                    (
                        "positions_and_reorder",
                        cp_size,
                        cp_rank,
                        packed_query_output_size,
                        packed_query_output_size,
                        packed_global_output_size,
                        query_cu_seqlens_cover_output,
                        key_cu_seqlens_cover_output,
                    ),
                    lambda: (
                        _cp_reorder_from_host(
                            host_cu_q,
                            host_cu_kv,
                            cp_size=cp_size,
                            cp_rank=cp_rank,
                            device=query.device,
                            local_output_size=packed_query_output_size,
                            key_local_output_size=packed_query_output_size,
                            global_output_size=packed_global_output_size,
                            query_cu_seqlens_cover_output=query_cu_seqlens_cover_output,
                            key_cu_seqlens_cover_output=key_cu_seqlens_cover_output,
                            return_inverse_key_reorder=True,
                        )
                        if host_cu_q is not None and host_cu_kv is not None
                        else dsa_layout.build_packed_allgather_cp_query_positions_and_key_reorder(
                            cu_seqlens_q=cu_seqlens_q,
                            cu_seqlens_kv=cu_seqlens_kv,
                            cp_size=cp_size,
                            cp_rank=cp_rank,
                            device=query.device,
                            local_output_size=packed_query_output_size,
                            key_local_output_size=packed_query_output_size,
                            global_output_size=packed_global_output_size,
                            query_cu_seqlens_cover_output=query_cu_seqlens_cover_output,
                            key_cu_seqlens_cover_output=key_cu_seqlens_cover_output,
                            return_inverse_key_reorder=True,
                        )
                    ),
                    prebuilt=_prebuilt_layout_tables(packed_query_output_size),
                )
            if packed_query_positions is not None:
                packed_query_positions = packed_query_positions.contiguous()
        if sequence_parallel_tp:
            if key.size(0) == local_sequence_rows:
                key = gather_from_sequence_parallel_region(key, group=tp_group)
            elif key.size(0) != sequence_parallel_tp_full_rows and not (
                reuse_latent_kv and cp_size > 1 and key.size(0) == cp_global_sequence_rows
            ):
                raise RuntimeError(
                    "DSA sequence-parallel key row count mismatch before CP gather: "
                    f"key_rows={key.size(0)}, local_rows={local_sequence_rows}, "
                    f"full_rows={sequence_parallel_tp_full_rows}, "
                    f"cp_global_rows={cp_global_sequence_rows}, tp_size={tp_size}, "
                    f"cp_size={cp_size}"
                )
            if value is not None:
                if value.size(0) == local_sequence_rows:
                    value = gather_from_sequence_parallel_region(value, group=tp_group)
                elif value.size(0) != sequence_parallel_tp_full_rows:
                    raise RuntimeError(
                        "DSA sequence-parallel value row count mismatch before CP gather: "
                        f"value_rows={value.size(0)}, local_rows={local_sequence_rows}, "
                        f"full_rows={sequence_parallel_tp_full_rows}, tp_size={tp_size}"
                    )

        if not packed_thd and cp_size > 1:
            _validate_nonpacked_cp_uniform_length(
                sq=cp_local_sequence_rows,
                skv=key.size(0),
                cp_size=cp_size,
                cp_group=cp_group,
                device=query.device,
            )

        local_cp_kv_lens = {cp_local_sequence_rows}
        local_cp_kv_len = None
        if cp_size > 1:
            assert (
                self.cp_comm_type == "allgather"
            ), "DSAttention context parallelism currently supports cp_comm_type=allgather only."

            # For allgather CP, keys/values are expected in full-sequence order.
            # Gather local-sequence tensors, then undo MCore's zigzag rank order.
            # Returns the reorder index and its inverse, which gives the reorder a gather
            # backward. The non-packed zigzag reorder is not memoized; its inverse is the
            # gathered key positions the builder already sorts, so it costs no extra kernel.
            def _build_kv_reorder_idx(local_len):
                if packed_thd:
                    # Same memo key shape as the branch above, so when the output sizes
                    # coincide this reuses that result instead of rerunning the cp_size loop.
                    _, idx, inv_idx = self._memoized(
                        layout_cache,
                        (
                            "positions_and_reorder",
                            cp_size,
                            cp_rank,
                            local_len,
                            local_len,
                            local_len * cp_size,
                            query_cu_seqlens_cover_output,
                            key_cu_seqlens_cover_output,
                        ),
                        lambda: (
                            _cp_reorder_from_host(
                                host_cu_q,
                                host_cu_kv,
                                cp_size=cp_size,
                                cp_rank=cp_rank,
                                device=query.device,
                                local_output_size=local_len,
                                key_local_output_size=local_len,
                                global_output_size=local_len * cp_size,
                                query_cu_seqlens_cover_output=query_cu_seqlens_cover_output,
                                key_cu_seqlens_cover_output=key_cu_seqlens_cover_output,
                                return_inverse_key_reorder=True,
                            )
                            if host_cu_q is not None and host_cu_kv is not None
                            else (
                                dsa_layout.build_packed_allgather_cp_query_positions_and_key_reorder
                            )(
                                cu_seqlens_q=cu_seqlens_q,
                                cu_seqlens_kv=cu_seqlens_kv,
                                cp_size=cp_size,
                                cp_rank=cp_rank,
                                device=query.device,
                                local_output_size=local_len,
                                key_local_output_size=local_len,
                                global_output_size=local_len * cp_size,
                                query_cu_seqlens_cover_output=query_cu_seqlens_cover_output,
                                key_cu_seqlens_cover_output=key_cu_seqlens_cover_output,
                                return_inverse_key_reorder=True,
                            )
                        ),
                        prebuilt=_prebuilt_layout_tables(local_len),
                    )
                    return idx, inv_idx
                return dsa_layout.build_zigzag_allgather_cp_key_reorder(
                    sq=local_len,
                    cp_size=cp_size,
                    device=query.device,
                    return_inverse_key_reorder=True,
                )

            gathered_cp_key = False
            gathered_cp_value = False
            if key.size(0) in local_cp_kv_lens:
                local_cp_kv_len = key.size(0)
                if kv_reorder_idx is None:
                    kv_reorder_idx, kv_reorder_inv_idx = _build_kv_reorder_idx(local_cp_kv_len)
                key = gather_from_sequence_parallel_region(key, group=cp_group)
                gathered_cp_key = True
            if value is not None and value.size(0) in local_cp_kv_lens:
                if local_cp_kv_len is None:
                    local_cp_kv_len = value.size(0)
                    if kv_reorder_idx is None:
                        kv_reorder_idx, kv_reorder_inv_idx = _build_kv_reorder_idx(local_cp_kv_len)
                elif value.size(0) != local_cp_kv_len:
                    raise RuntimeError(
                        "DSA local key/value sequence length mismatch before CP gather: "
                        f"key_len={local_cp_kv_len}, value_len={value.size(0)}"
                    )
                value = gather_from_sequence_parallel_region(value, group=cp_group)
                gathered_cp_value = True
            if kv_reorder_idx is not None:
                if gathered_cp_key:
                    if key.size(0) != kv_reorder_idx.numel():
                        raise RuntimeError(
                            "DSA gathered key length mismatch: "
                            f"key_seqlen={key.size(0)}, expected={kv_reorder_idx.numel()}"
                        )
                    key = _permute_rows(key, kv_reorder_idx, kv_reorder_inv_idx)
                if gathered_cp_value:
                    if value.size(0) != kv_reorder_idx.numel():
                        raise RuntimeError(
                            "DSA gathered value length mismatch: "
                            f"value_seqlen={value.size(0)}, expected={kv_reorder_idx.numel()}"
                        )
                    value = _permute_rows(value, kv_reorder_idx, kv_reorder_inv_idx)

        skv = key.size(0)

        if self.mtp_latent_kv_share and mtp_payload is not None and is_mtp_source:
            # Capture the exact canonical tensor consumed by sparse attention: TP/SP and
            # CP gathers plus CP reorder have already run. This must precede the combined
            # fused DSA early return, which remains eligible for latent-KV-only sharing.
            assert mtp_payload is not None
            mtp_payload.latent_kv_by_layer[self.layer_number] = key

        if not packed_thd and sequence_parallel_query_is_local:
            nonpacked_query_positions = dsa_layout.extract_query_positions_from_position_ids(
                position_ids, sq, query.device
            )
            if nonpacked_query_positions is None:
                full_query_positions, _ = dsa_layout.get_cp_positions_from_layout(
                    sq=sequence_parallel_tp_full_rows,
                    skv=skv,
                    cp_size=cp_size,
                    cp_rank=cp_rank,
                    cp_comm_type=self.cp_comm_type,
                    device=query.device,
                    cp_group=cp_group,
                )
                row_start = sequence_parallel_tp_row_start
                nonpacked_query_positions = full_query_positions[
                    row_start : row_start + sq
                ].contiguous()

        # Detach x and qr to prevent gradients of indexer from flowing back to the main model.
        x = x.detach()
        qr = qr.detach()

        indexer_loss_coeff = self.config.dsa_indexer_loss_coeff or 0.0
        computes_topk = not self.skip_topk and not reuse_sparse_attention_index
        use_indexer_loss = (
            self.training and torch.is_grad_enabled() and indexer_loss_coeff > 0 and computes_topk
        )
        _is_kpool = self.indexer is not None and getattr(self.indexer, "index_kpool", 1) > 1
        # A KPool indexer with a positive loss coefficient is trainable. Its layer selects
        # through the trainable path (pool, then gather over CP) in every forward: training,
        # eval and no_grad alike, so its pools never depend on whether the loss runs;
        # use_indexer_loss only decides whether the teacher and the loss are computed.
        kpool_trainable = _is_kpool and indexer_loss_coeff > 0 and computes_topk
        if use_indexer_loss and sequence_parallel_query_is_local:
            raise RuntimeError(
                "DSA indexer loss requires TP ranks to own the same query rows; "
                "sequence-local TP query shards cannot form a global-head target."
            )
        float_mask, varlen_params, varlen_is_plain_causal = (
            dsa_masking.build_dsattention_forward_mask(
                sq=sq,
                skv=skv,
                b=b,
                device=x.device,
                cp_size=cp_size,
                cp_rank=cp_rank,
                cp_comm_type=self.cp_comm_type,
                cp_group=cp_group,
                attn_mask_type=attn_mask_type,
                attention_mask=attention_mask,
                position_ids=position_ids,
                packed_seq_params=packed_seq_params,
                packed_query_positions=packed_query_positions,
                nonpacked_query_positions=nonpacked_query_positions,
            )
        )
        if varlen_params is not None:
            varlen_starts, varlen_ends, key_positions = varlen_params
        else:
            varlen_starts = varlen_ends = key_positions = None
        query_valid_rows = dsa_masking.extract_query_valid_rows_from_packed_seq_params(
            packed_seq_params, b=b, sq=sq, device=query.device
        )
        if kpool_trainable:
            # Layouts the trainable KPool path does not cover. The config cannot see them:
            # they come with the packed parameters and the mask of each forward.
            kpool_size = self.indexer.index_kpool
            if packed_thd and cp_size > 1:
                raise ValueError(
                    "A trainable KPool indexer (dsa_indexer_loss_coeff > 0) does not support "
                    "packed THD sequences with context parallelism: its pools restart at every "
                    "document, which would span CP ranks. Use cp_size=1 for packed sequences."
                )
            if float_mask is not None:
                raise ValueError(
                    "A trainable KPool indexer (dsa_indexer_loss_coeff > 0) does not support an "
                    "explicit attention mask; use the causal mask type."
                )
            if cp_size > 1 and cp_local_sequence_rows % (2 * kpool_size) != 0:
                raise ValueError(
                    "A trainable KPool indexer (dsa_indexer_loss_coeff > 0) under context "
                    "parallelism pools each CP rank's tokens before the all-gather, so each "
                    "zigzag half must hold whole pools: the local sequence length must be a "
                    f"multiple of 2 * dsa_indexer_kpool = {2 * kpool_size}, got "
                    f"{cp_local_sequence_rows}."
                )
        use_fused_kernels = dsa_kernels.use_fused_dsa_kernels(self.config)
        sparse_indexer_loss = self.config.dsa_indexer_use_sparse_loss
        # Reports a packed causal layout with identity key positions. CP size is an
        # independent geometry fact; the scoring plan and packed metadata eligibility
        # decide which executor can safely consume the layout.
        use_local_indexer_varlen = (
            packed_thd
            and attn_mask_type == AttnMaskType.causal
            and varlen_starts is not None
            and varlen_ends is not None
            and key_positions is None
        )
        indexer_reduce_group = (
            cp_group if cp_size > 1 and self.config.calculate_per_token_loss else None
        )
        indexer_avg_group = (
            cp_group if cp_size > 1 and not self.config.calculate_per_token_loss else None
        )

        index_payload = (
            self._get_dsa_index_sharing_payload(packed_seq_params, attention_mask)
            if self.index_share
            else None
        )
        # Under activation recompute the checkpoint's first call runs without grad, so it
        # selects top-k without the indexer loss, and the replay reruns the dense indexer
        # and top-k only to recover the selected scores for that loss. With
        # dsa_indexer_reuse_checkpoint_topk the first call also keeps those scores next to
        # its published indices, and the replay computes the loss from them. A frozen
        # indexer (no indexer loss) needs no scores: its replay reuses the top-k that the
        # first call published and skips the indexer entirely.
        indexer_loss_on = indexer_loss_coeff > 0
        reuse_checkpoint_topk = bool(
            self.config.dsa_indexer_reuse_checkpoint_topk
            and index_payload is not None
            and computes_topk
            and self.training
            and (sparse_indexer_loss if indexer_loss_on else not _is_kpool)
            # KPool never reuses a checkpoint's top-k, frozen or trainable; the config
            # rejects the combination, and this keeps a later-set flag from opening it.
            and not _is_kpool
            and use_fused_kernels
            and not sequence_parallel_query_is_local
        )
        checkpoint_phase = (
            self._get_forward_sharing_state(packed_seq_params, attention_mask).checkpoint_phase
            if reuse_checkpoint_topk
            else CheckpointPhase.NONE
        )
        stash_checkpoint_topk_scores = (
            checkpoint_phase is CheckpointPhase.FIRST_FORWARD
            and not torch.is_grad_enabled()
            and indexer_loss_on
        )
        replay_checkpoint_topk = checkpoint_phase is CheckpointPhase.REPLAY and use_indexer_loss
        reuse_checkpoint_indices = False
        if checkpoint_phase is CheckpointPhase.REPLAY and not indexer_loss_on:
            # The replay's snapshot was taken right after the first call, so this layer's
            # entry is the selection that the first call's output already used, as long as
            # the checkpoint ran the layer once.
            assert index_payload is not None
            replayed_layers = self._get_forward_sharing_state(
                packed_seq_params, attention_mask
            ).get_or_create(_DSAFrozenReplayPayload)
            if self.layer_number in replayed_layers.layer_numbers:
                raise RuntimeError(
                    "DSA checkpoint replay ran layer_number="
                    f"{self.layer_number} twice: dsa_indexer_reuse_checkpoint_topk keeps one "
                    "first-call top-k per layer and checkpoint, so a checkpoint must not run "
                    "a layer more than once."
                )
            replayed_layers.layer_numbers.add(self.layer_number)
            stashed_indices = index_payload.topk_by_layer.get(self.layer_number)
            stashed_key_len = index_payload.frozen_topk_key_len_by_layer.get(self.layer_number)
            if stashed_indices is None:
                _warn_dsa_checkpoint_topk_reuse_fallback(
                    self.layer_number, "its first call published no top-k indices"
                )
            elif stashed_key_len != skv or tuple(stashed_indices.shape[:2]) != (b, sq):
                raise RuntimeError(
                    "DSA checkpoint replay geometry differs from its first call: "
                    f"layer_number={self.layer_number}, stored key length {stashed_key_len} "
                    f"and rows {tuple(stashed_indices.shape[:2])}, replay key length {skv} "
                    f"and rows {(b, sq)}."
                )
            else:
                reuse_checkpoint_indices = True
        topk_indices = None
        topk_length = None
        topk_scores = None
        q = k = weights = None
        # Pooled indexer keys of the trainable KPool path, in global pool order, and each
        # pool's first token.
        kpool_pooled_keys = kpool_pool_token_base = None
        local_packed_cp_query_start = 0
        local_packed_cp_query_len = sq
        if sequence_parallel_query_is_local:
            local_packed_cp_query_start = sequence_parallel_tp_row_start
            local_packed_cp_query_len = sequence_parallel_tp_full_rows

        shared_topk_rows_nonempty = False
        if reuse_sparse_attention_index:
            assert mtp_payload is not None
            topk_indices = mtp_payload.topk_by_layer[self.layer_number]
            topk_length = mtp_payload.topk_length_by_layer.get(self.layer_number)
            shared_topk_rows_nonempty = mtp_payload.topk_rows_nonempty_by_layer.get(
                self.layer_number, False
            )
        elif self.skip_topk:
            assert index_payload is not None
            if self.source_layer not in index_payload.topk_by_layer:
                raise RuntimeError(
                    "DSA index-share skip layer "
                    f"(layer_number={self.layer_number}) needs top-k indices from source "
                    f"computing layer {self.source_layer}, but that layer did not run before it "
                    "in this pipeline stage. Cross-PP top-k sharing is not supported. Ensure each "
                    "pipeline stage starts on a computing layer "
                    f"(dsa_indexer_topk_freq={self.index_topk_freq}, "
                    f"dsa_indexer_skip_topk_offset={self.index_skip_topk_offset}). "
                    f"Index sharing payload has layers {sorted(index_payload.topk_by_layer)}."
                )
            topk_indices = index_payload.topk_by_layer[self.source_layer]
            topk_length = index_payload.topk_length_by_layer.get(self.source_layer)
            shared_topk_rows_nonempty = index_payload.topk_rows_nonempty_by_layer.get(
                self.source_layer, False
            )
        elif reuse_checkpoint_indices:
            assert index_payload is not None
            topk_indices = index_payload.topk_by_layer[self.layer_number]
            topk_length = index_payload.topk_length_by_layer.get(self.layer_number)
            shared_topk_rows_nonempty = index_payload.topk_rows_nonempty_by_layer.get(
                self.layer_number, False
            )
        else:
            assert self.indexer is not None
            with torch.enable_grad() if use_indexer_loss else torch.no_grad():
                if kpool_trainable:
                    # The trainable path pools this rank's tokens and gathers the pooled
                    # keys over CP; the token-level indexer K and the gate stay local.
                    q, k, weights, kpool_gate_score = self.indexer.forward_before_topk(
                        x, qr, packed_seq_params, return_kpool_gate_score=True
                    )
                    kpool_pooled_keys, kpool_pool_token_base = self._kpool_pool_then_gather(
                        k,
                        kpool_gate_score,
                        cu_seqlens_kv=cu_seqlens_kv if packed_thd else None,
                        cp_size=cp_size,
                        cp_group=cp_group,
                        local_rows=cp_local_sequence_rows,
                    )
                    del kpool_gate_score
                else:
                    q, k, weights = self.indexer.forward_before_topk(x, qr, packed_seq_params)
                if cp_size > 1 and not kpool_trainable and k.size(0) in local_cp_kv_lens:
                    if kv_reorder_idx is None:
                        kv_reorder_idx, kv_reorder_inv_idx = _build_kv_reorder_idx(k.size(0))
                    k = gather_from_sequence_parallel_region(k, group=cp_group)
                    if k.size(0) != kv_reorder_idx.numel():
                        raise RuntimeError(
                            "DSA gathered indexer-key length mismatch: "
                            f"k_seqlen={k.size(0)}, expected={kv_reorder_idx.numel()}"
                        )
                    k = _permute_rows(k, kv_reorder_idx, kv_reorder_inv_idx)
                    # Apply the same CP gather + reorder to the kpool gate score so it
                    # matches the now-global key length (otherwise the per-seg reshape
                    # in _kpool_compress_keys crashes with a size mismatch).
                    gate = self.indexer._kpool_gate_score
                    if gate is not None and gate.size(0) in local_cp_kv_lens:
                        gate = gather_from_sequence_parallel_region(gate, group=cp_group)
                        gate = gate.index_select(0, kv_reorder_idx)
                        self.indexer._kpool_gate_score = gate
                if sequence_parallel_tp and q.size(0) != sq:
                    if (
                        q.size(0) != sequence_parallel_tp_full_rows
                        or weights.size(0) != sequence_parallel_tp_full_rows
                    ):
                        raise RuntimeError(
                            "DSA sequence-parallel indexer row count mismatch: "
                            f"q_rows={q.size(0)}, weights_rows={weights.size(0)}, "
                            f"query_rows={sq}, full_rows={sequence_parallel_tp_full_rows}, "
                            f"tp_size={tp_size}"
                        )
                    if not sequence_parallel_query_is_local:
                        raise RuntimeError(
                            "DSA indexer produced TP-gathered rows while attention query rows "
                            "were not sequence-local."
                        )
                    row_start = sequence_parallel_tp_row_start
                    row_end = row_start + sq
                    q = q[row_start:row_end].contiguous()
                    weights = weights[row_start:row_end].contiguous()

        def compute_indexer_loss_with_reference_path():
            # The token-level loss selects tokens, not pools; KPool layers take the
            # KPoolSparseIndexerLoss branch below and must never reach this one.
            assert not _is_kpool, "KPool layers must not reach the token-level indexer loss."
            key_for_loss = key.detach()
            if absorbed_mla and key_for_loss.size(2) == 1 and query.size(2) > 1:
                key_for_loss = key_for_loss.expand(-1, -1, query.size(2), -1)
            return FusedDSAIndexerLoss.apply(
                q,
                weights,
                k,
                query.detach(),
                key_for_loss,
                self.softmax_scale,
                self.index_topk,
                indexer_loss_coeff,
                float_mask,
                sparse_indexer_loss,
                self.pg_collection,
                varlen_starts,
                varlen_ends,
                key_positions,
                query_valid_rows,
                self.config.calculate_per_token_loss,
                self.config.dsa_indexer_scoring_relu,
            )

        fused_output = None
        # kpool DSA indexer (GLM-5.3-Flash, index_kpool > 1): the fused cuDNN DSA
        # path computes its own per-token top-k and does NOT implement pool-granular
        # selection / key compression / tail-append. Bypass it so the Python kpool
        # top-k path below runs instead (mirrors DSAIndexer.forward_with_scores).
        if (
            use_fused_kernels
            and not self.index_share
            and not self.mtp_sparse_attention_index_share
            and not _is_kpool
        ):
            assert q is not None and k is not None and weights is not None
            fused_output = dsa_kernels.run_fused_dsa_attention(
                config=self.config,
                query=query,
                key=key,
                value=value,
                up_v_weight=up_v_weight,
                q_indexer=q,
                k_indexer=k,
                indexer_weights=weights,
                indexer_topk=self.index_topk,
                softmax_scale=self.softmax_scale,
                loss_coeff=indexer_loss_coeff if use_indexer_loss else 0.0,
                sparse_loss=sparse_indexer_loss,
                calculate_per_token_loss=self.config.calculate_per_token_loss,
                absorbed_mla=absorbed_mla,
                cp_size=cp_size,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
                varlen_starts=varlen_starts,
                varlen_ends=varlen_ends,
                key_positions=key_positions,
                query_valid_rows=query_valid_rows,
                varlen_is_plain_causal=varlen_is_plain_causal,
                use_relu=self.config.dsa_indexer_scoring_relu,
                use_local_indexer_varlen=use_local_indexer_varlen,
                single_packed_thd_sequence=single_packed_thd_sequence,
                local_packed_cp_rank=cp_rank,
                local_packed_cp_query_start=local_packed_cp_query_start,
                local_packed_cp_query_len=local_packed_cp_query_len,
                pg_collection=self.pg_collection,
            )
        if fused_output is not None:
            output, indexer_loss = fused_output
            if use_indexer_loss:
                if indexer_loss is None:
                    raise RuntimeError("Fused DSA attention did not produce a valid indexer loss.")
                DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                    loss=indexer_loss,
                    raw_loss=indexer_loss / indexer_loss_coeff,
                    layer_number=self.layer_number,
                    num_layers=max(
                        self.layer_number,
                        self.config.num_layers + (self.config.mtp_num_layers or 0),
                    ),
                    reduce_group=indexer_reduce_group,
                    avg_group=indexer_avg_group,
                )
                output = DSAIndexerLossAutoScaler.apply(output, indexer_loss)
            return _normalize_dsattention_output_rank(output, x.ndim)

        fused_bounds = None
        if use_fused_kernels and computes_topk and topk_indices is None:
            assert q is not None
            fused_bounds = dsa_masking.build_fused_indexer_varlen_bounds(
                sq=sq,
                skv=skv,
                device=q.device,
                mask=float_mask,
                varlen_starts=varlen_starts,
                varlen_ends=varlen_ends,
                key_positions=key_positions,
            )

        indexer_loss = None

        def reject_mxfp8_reference_fallback(what: str) -> None:
            if not use_fused_kernels:
                reason = "the fused DSA kernels are disabled"
            elif fused_bounds is None:
                reason = "no fused indexer bounds exist for this attention mask layout"
            else:
                reason = (
                    "the cuDNN indexer hook declined this layout (DEBUG logging of "
                    "dsa_kernels names the hook)"
                )
            _reject_mxfp8_indexer_reference_fallback(
                self.config, f"the {what} of DSA layer {self.layer_number}", reason
            )

        def slice_topk_to_local_sequence_parallel_rows():
            nonlocal topk_indices, topk_length
            if topk_indices is None or not sequence_parallel_query_is_local:
                return
            topk_sq = topk_indices.size(1)
            if topk_sq == sq:
                return
            expected_topk_sq = sequence_parallel_tp_full_rows
            if topk_sq != expected_topk_sq:
                raise RuntimeError(
                    "DSA sequence-parallel top-k row count mismatch: "
                    f"topk_rows={topk_sq}, query_rows={sq}, tp_size={tp_size}"
                )
            row_start = sequence_parallel_tp_row_start
            row_end = row_start + sq
            topk_indices = topk_indices[:, row_start:row_end].contiguous()
            if topk_length is not None:
                topk_length = topk_length[:, row_start:row_end].contiguous()

        def kpool_selector_inputs(pool_scores_out=None):
            """The mxfp8 guard and the pool scorer shared by both KPool selection branches."""
            _reject_mxfp8_indexer_reference_fallback(
                self.config,
                f"the KPool indexer top-k of DSA layer {self.layer_number}",
                "KPool selection does not use the cuDNN dense scorer",
                path="the KPool selector",
            )
            return self._kpool_pool_topk_fn(
                packed_thd=packed_thd,
                has_mask=float_mask is not None,
                positions_from_position_ids=position_ids is not None
                and (cp_size > 1 or sequence_parallel_query_is_local),
                local_rows=sq,
                full_rows=sequence_parallel_tp_full_rows,
                row_start=(
                    sequence_parallel_tp_row_start if sequence_parallel_query_is_local else 0
                ),
                skv=skv,
                cp_size=cp_size,
                cp_rank=cp_rank,
                pool_scores_out=pool_scores_out,
            )

        # The trainable KPool loss of this forward runs on the cuDNN frontend kernels after the
        # sparse attention, whose LSE its teacher reads (kpool_cudnn_loss), or else in PyTorch
        # (KPoolSparseIndexerLoss) before it.
        kpool_cudnn_loss = False
        kpool_pool_scores = None
        kpool_metric_num_layers = max(
            self.layer_number, self.config.num_layers + (self.config.mtp_num_layers or 0)
        )
        if kpool_trainable:
            assert q is not None and weights is not None and kpool_pooled_keys is not None
            # ===================================
            # Trainable KPool: select pools, then attach the sparse pool-level loss
            # ===================================
            kpool_size = self.indexer.index_kpool
            # Decided per forward from host facts, then from whether the cuDNN KPool scorer
            # accepts it: its pool scores are the cuDNN loss's student.
            kpool_pytorch_loss_reason = None
            kpool_pool_scores_out = None
            if use_indexer_loss:
                kpool_pytorch_loss_reason = self._kpool_cudnn_loss_decline_reason(
                    tp_size=tp_size, absorbed_mla=absorbed_mla, query=query, key=key
                )
                if kpool_pytorch_loss_reason is None:
                    kpool_pool_scores_out = {}
            kpool_pool_topk_fn = kpool_selector_inputs(pool_scores_out=kpool_pool_scores_out)
            # Selection is discrete: it runs without grad on detached inputs, from the pooled
            # keys built above (bit for bit the pools the frozen path compresses).
            with torch.no_grad():
                _index_scores, topk_indices, kpool_pool_ids = fused_qk_topk_kpool(
                    q.detach(),
                    k.detach(),
                    weights.detach(),
                    self.index_topk,
                    kpool_size,
                    None,
                    self.indexer.index_kpool_compress_ape,
                    mask=float_mask,
                    varlen_starts=varlen_starts,
                    varlen_ends=varlen_ends,
                    key_positions=key_positions,
                    cu_seqlens_kv=cu_seqlens_kv if packed_thd else None,
                    use_relu=self.config.dsa_indexer_scoring_relu,
                    always_select_tail=self.indexer.index_kpool_always_select_tail,
                    fp8_indexer=self.indexer.index_kpool_use_quantization,
                    rotate_activation_enabled=self.config.dsa_indexer_rotate_activation,
                    return_index_scores=False,
                    kpool_key_dtype_compression=(
                        self.config.dsa_indexer_kpool_key_dtype_compression
                    ),
                    hf_scoring=self.config.dsa_indexer_hf_scoring,
                    pool_topk_fn=kpool_pool_topk_fn,
                    pooled_keys=(kpool_pooled_keys.detach(), kpool_pool_token_base),
                    return_selection=True,
                )
                del _index_scores
            if tp_size > 1 and not sequence_parallel_query_is_local:
                # TP ranks select on their own, and the cuDNN top-k (or an exact tie at the
                # selection boundary) need not agree across them. Rank 0's selection then
                # feeds every rank's sparse attention and teacher, so all TP ranks attend
                # the same tokens and compute the same TP-averaged indexer loss.
                topk_indices, kpool_pool_ids = self._broadcast_kpool_selection_from_tp_rank0(
                    topk_indices, kpool_pool_ids, tp_group
                )
            slice_topk_to_local_sequence_parallel_rows()
            if use_indexer_loss:
                if kpool_pytorch_loss_reason is None:
                    kpool_pool_scores = kpool_pool_scores_out.get("pool_scores")
                    if kpool_pool_scores is None:
                        kpool_pytorch_loss_reason = (
                            "the cuDNN KPool scorer declined this forward (packed THD, an "
                            "explicit mask, position_ids-derived positions, non-uniform CP or "
                            "unsupported inputs; its own warning names which)"
                        )
                kpool_cudnn_loss = kpool_pytorch_loss_reason is None
                _log_kpool_loss_path_once(self.config, kpool_pytorch_loss_reason)
            if use_indexer_loss and not kpool_cudnn_loss:
                # Also records the layer's tail mass p_E, in its own tracker slots.
                indexer_loss = self._kpool_sparse_indexer_loss(
                    q=q,
                    weights=weights,
                    k_pooled=kpool_pooled_keys,
                    topk_indices=topk_indices,
                    pool_ids=kpool_pool_ids,
                    query=query,
                    key=key,
                    query_valid_rows=query_valid_rows,
                    indexer_loss_coeff=indexer_loss_coeff,
                    tp_group=tp_group,
                    metric_num_layers=kpool_metric_num_layers,
                    metric_reduce_group=indexer_reduce_group,
                    metric_avg_group=indexer_avg_group,
                )
                DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                    loss=indexer_loss,
                    raw_loss=indexer_loss / indexer_loss_coeff,
                    layer_number=self.layer_number,
                    num_layers=kpool_metric_num_layers,
                    reduce_group=indexer_reduce_group,
                    avg_group=indexer_avg_group,
                )
        elif use_indexer_loss:
            assert q is not None and k is not None and weights is not None
            # ===================================
            # Attach indexer topk and loss
            # ===================================
            if replay_checkpoint_topk:
                assert index_payload is not None
                stashed_indices = index_payload.topk_by_layer.get(self.layer_number)
                stashed_length = index_payload.topk_length_by_layer.get(self.layer_number)
                stashed_scores = index_payload.topk_scores_by_layer.get(self.layer_number)
                stashed_key_len = index_payload.topk_key_len_by_layer.get(self.layer_number)
                reuse_failure = None
                if stashed_indices is None or stashed_length is None or stashed_scores is None:
                    reuse_failure = "its first call stored no top-k scores"
                else:
                    if stashed_key_len != skv or tuple(stashed_indices.shape[:2]) != (b, sq):
                        raise RuntimeError(
                            "DSA checkpoint replay geometry differs from its first call: "
                            f"layer_number={self.layer_number}, stored key length "
                            f"{stashed_key_len} and rows {tuple(stashed_indices.shape[:2])}, "
                            f"replay key length {skv} and rows {(b, sq)}."
                        )
                    stashed_loss = dsa_kernels.run_sparse_indexer_loss_from_topk(
                        self.config,
                        q,
                        k,
                        weights,
                        topk_indices=stashed_indices,
                        topk_length=stashed_length,
                        topk_scores=stashed_scores,
                        query=query.detach(),
                        key=key.detach(),
                        softmax_scale=self.softmax_scale,
                        loss_coeff=indexer_loss_coeff,
                        pg_collection=self.pg_collection,
                        query_valid_rows=query_valid_rows,
                        calculate_per_token_loss=self.config.calculate_per_token_loss,
                    )
                    if stashed_loss is None:
                        reuse_failure = "the kernel backend declined the stored top-k"
                    else:
                        topk_indices = stashed_indices
                        topk_length = stashed_length
                        topk_scores = stashed_scores
                        indexer_loss = stashed_loss
                if reuse_failure is not None:
                    _warn_dsa_checkpoint_topk_reuse_fallback(self.layer_number, reuse_failure)

            if sparse_indexer_loss and fused_bounds is not None and topk_indices is None:
                starts_i32, ends_i32 = fused_bounds
                block_size = int(getattr(self, "fused_indexer_block_size", 8192))
                fused_topk_with_loss = dsa_kernels.run_fused_qk_topk_with_loss(
                    self.config,
                    q,
                    k,
                    weights,
                    self.index_topk,
                    starts_i32,
                    ends_i32,
                    block_size=max(1, block_size),
                    query=query.detach(),
                    key=key.detach(),
                    softmax_scale=self.softmax_scale,
                    loss_coeff=indexer_loss_coeff,
                    pg_collection=self.pg_collection,
                    query_valid_rows=query_valid_rows,
                    calculate_per_token_loss=self.config.calculate_per_token_loss,
                    use_relu=self.config.dsa_indexer_scoring_relu,
                    use_local_indexer_varlen=use_local_indexer_varlen,
                    single_packed_thd_sequence=single_packed_thd_sequence,
                    local_packed_cp_rank=cp_rank,
                    local_packed_cp_query_start=local_packed_cp_query_start,
                    local_packed_cp_query_len=local_packed_cp_query_len,
                    packed_seq_params=packed_seq_params,
                    cp_size=cp_size,
                    varlen_is_plain_causal=varlen_is_plain_causal,
                )
                if fused_topk_with_loss is not None:
                    topk_indices, topk_length, indexer_loss = fused_topk_with_loss

            if topk_indices is None or indexer_loss is None:
                reject_mxfp8_reference_fallback("indexer top-k and loss")
                topk_indices, indexer_loss = compute_indexer_loss_with_reference_path()
            # No TP-local top-k slicing here: the guard above forbids the indexer loss
            # under sequence-local TP query shards, so the top-k rows are already global.

            # Save indexer loss for logging.
            if indexer_loss_coeff > 0:
                DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                    loss=indexer_loss,
                    raw_loss=indexer_loss / indexer_loss_coeff,
                    layer_number=self.layer_number,
                    num_layers=max(
                        self.layer_number,
                        self.config.num_layers + (self.config.mtp_num_layers or 0),
                    ),
                    reduce_group=indexer_reduce_group,
                    avg_group=indexer_avg_group,
                )
        elif topk_indices is None:
            assert q is not None and k is not None and weights is not None
            # ===================================
            # Get top-k indices
            # ===================================
            if _is_kpool:
                kpool_pool_topk_fn = kpool_selector_inputs()
                # KPool selection is discrete and does not need an autograd graph.
                with torch.no_grad():
                    _index_scores, topk_indices = fused_qk_topk_kpool(
                        q,
                        k,
                        weights,
                        self.index_topk,
                        self.indexer.index_kpool,
                        self.indexer._kpool_gate_score,
                        self.indexer.index_kpool_compress_ape,
                        mask=float_mask,
                        varlen_starts=varlen_starts,
                        varlen_ends=varlen_ends,
                        key_positions=key_positions,
                        cu_seqlens_kv=cu_seqlens_kv if packed_thd else None,
                        use_relu=self.config.dsa_indexer_scoring_relu,
                        always_select_tail=self.indexer.index_kpool_always_select_tail,
                        fp8_indexer=self.indexer.index_kpool_use_quantization,
                        rotate_activation_enabled=self.config.dsa_indexer_rotate_activation,
                        return_index_scores=False,
                        kpool_key_dtype_compression=(
                            self.config.dsa_indexer_kpool_key_dtype_compression
                        ),
                        hf_scoring=self.config.dsa_indexer_hf_scoring,
                        pool_topk_fn=kpool_pool_topk_fn,
                    )
                    del _index_scores
                    self.indexer._kpool_gate_score = None
            elif fused_bounds is not None:
                starts_i32, ends_i32 = fused_bounds
                block_size = int(getattr(self, "fused_indexer_block_size", 8192))
                fused_topk_args = (
                    self.config,
                    q,
                    k,
                    weights,
                    self.index_topk,
                    starts_i32,
                    ends_i32,
                )
                fused_topk_kwargs = dict(
                    block_size=max(1, block_size),
                    use_relu=self.config.dsa_indexer_scoring_relu,
                    use_local_indexer_varlen=use_local_indexer_varlen,
                    single_packed_thd_sequence=single_packed_thd_sequence,
                    local_packed_cp_rank=cp_rank,
                    local_packed_cp_query_start=local_packed_cp_query_start,
                    local_packed_cp_query_len=local_packed_cp_query_len,
                    packed_seq_params=packed_seq_params,
                    cp_size=cp_size,
                    varlen_is_plain_causal=varlen_is_plain_causal,
                )
                if stash_checkpoint_topk_scores:
                    fused_topk_with_scores = dsa_kernels.run_fused_qk_topk_with_scores(
                        *fused_topk_args, query=query, key=key, **fused_topk_kwargs
                    )
                    if fused_topk_with_scores is not None:
                        topk_indices, topk_length, topk_scores = fused_topk_with_scores
                if topk_indices is None:
                    fused_topk = dsa_kernels.run_fused_qk_topk(
                        *fused_topk_args, **fused_topk_kwargs
                    )
                    if fused_topk is not None:
                        topk_indices, topk_length = fused_topk

            if topk_indices is None:
                reject_mxfp8_reference_fallback("indexer top-k")
                with torch.no_grad():
                    index_scores, topk_indices = fused_qk_topk_naive(
                        q,
                        k,
                        weights,
                        self.index_topk,
                        mask=float_mask,
                        varlen_starts=varlen_starts,
                        varlen_ends=varlen_ends,
                        key_positions=key_positions,
                        use_relu=self.config.dsa_indexer_scoring_relu,
                    )
                    del index_scores
            slice_topk_to_local_sequence_parallel_rows()

        # Host-side certificate that every top-k row selects a key, which lets the sparse
        # attention backend skip its empty-row masking. A computing layer proves it from its
        # layout. Index-sharing and repeated-MTP consumers reuse the producer's exact top-k
        # tensors, from which alone the backend finds empty rows, so they inherit the
        # producer's published certificate; their own layout must still pass the same checks.
        # The helper's computes_topk gate thus also admits a certified producer's top-k.
        # KPool selection without its tail leaves the rows before a query's first complete
        # pool without a key, so such a producer certifies (and publishes) nothing.
        kpool_may_leave_rows_empty = _is_kpool and not self.indexer.index_kpool_always_select_tail
        topk_has_certified_source = (
            computes_topk and not kpool_may_leave_rows_empty
        ) or shared_topk_rows_nonempty
        all_topk_rows_nonempty = dsa_masking.can_prove_all_topk_rows_nonempty(
            computes_topk=topk_has_certified_source,
            indexer_topk=self.index_topk,
            kv_sequence_length=skv,
            attention_mask=float_mask,
            query_valid_rows=query_valid_rows,
            varlen_is_plain_causal=varlen_is_plain_causal,
            use_local_indexer_varlen=use_local_indexer_varlen,
            varlen_starts=varlen_starts,
            varlen_ends=varlen_ends,
            key_positions=key_positions,
        )

        publish_ordinary_topk = self.index_share and computes_topk
        publish_mtp_topk = bool(
            self.mtp_sparse_attention_index_share and mtp_payload is not None and is_mtp_source
        )
        # Every publication writes the certificate, False included, so a stale True can
        # never describe newer indices.
        if publish_ordinary_topk:
            assert index_payload is not None and topk_indices is not None
            index_payload.topk_by_layer[self.layer_number] = topk_indices
            index_payload.topk_rows_nonempty_by_layer[self.layer_number] = all_topk_rows_nonempty
            if topk_length is None:
                index_payload.topk_length_by_layer.pop(self.layer_number, None)
            else:
                index_payload.topk_length_by_layer[self.layer_number] = topk_length
            if topk_scores is None:
                index_payload.topk_scores_by_layer.pop(self.layer_number, None)
                index_payload.topk_key_len_by_layer.pop(self.layer_number, None)
            else:
                index_payload.topk_scores_by_layer[self.layer_number] = topk_scores
                index_payload.topk_key_len_by_layer[self.layer_number] = skv
            if not indexer_loss_on:
                index_payload.frozen_topk_key_len_by_layer[self.layer_number] = skv
        if publish_mtp_topk:
            assert mtp_payload is not None and topk_indices is not None
            mtp_payload.topk_by_layer[self.layer_number] = topk_indices
            mtp_payload.topk_rows_nonempty_by_layer[self.layer_number] = all_topk_rows_nonempty
            if topk_length is not None:
                mtp_payload.topk_length_by_layer[self.layer_number] = topk_length

        # ===================================
        # Run sparse attention kernel
        # ===================================
        output = _run_sparse_attention(
            absorbed_mla=absorbed_mla,
            query=query,
            key=key,
            value=value,
            up_v_weight=up_v_weight,
            topk_indices=topk_indices,
            topk_length=topk_length,
            softmax_scale=self.softmax_scale,
            config=self.config,
            mask=float_mask,
            varlen_starts=varlen_starts,
            varlen_ends=varlen_ends,
            key_positions=key_positions,
            all_topk_rows_nonempty=all_topk_rows_nonempty,
            # The cuDNN KPool teacher reads the attention's own LSE; asked for only then, so
            # every other call is unchanged. The output is the same either way.
            **({"return_lse": True} if kpool_cudnn_loss else {}),
        )

        if kpool_cudnn_loss:
            output, attention_lse = output
            # Also records the layer's tail mass p_E, in its own tracker slots.
            indexer_loss = self._kpool_sparse_indexer_loss_cudnn(
                q=q,
                weights=weights,
                k_pooled=kpool_pooled_keys,
                topk_indices=topk_indices,
                pool_ids=kpool_pool_ids,
                pool_scores=kpool_pool_scores,
                query=query,
                key=key,
                lse=attention_lse,
                query_valid_rows=query_valid_rows,
                indexer_loss_coeff=indexer_loss_coeff,
                metric_num_layers=kpool_metric_num_layers,
                metric_reduce_group=indexer_reduce_group,
                metric_avg_group=indexer_avg_group,
            )
            del attention_lse
            DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                loss=indexer_loss,
                raw_loss=indexer_loss / indexer_loss_coeff,
                layer_number=self.layer_number,
                num_layers=kpool_metric_num_layers,
                reduce_group=indexer_reduce_group,
                avg_group=indexer_avg_group,
            )

        if use_indexer_loss:
            if indexer_loss is None:
                raise RuntimeError("Indexer loss path did not produce a valid loss tensor.")
            output = DSAIndexerLossAutoScaler.apply(output, indexer_loss)

        return _normalize_dsattention_output_rank(output, x.ndim)
