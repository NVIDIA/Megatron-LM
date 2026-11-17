# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Batch-invariant simplified DSA-GQA: one routing and attention path for training, prefill
and decode, built on the kernels in ``dsa_gqa_inference_kernels.py``.
Don't import this file if batch-invariant mode is not supported as it will error out
that deep_select is not installed.
"""

import torch
import torch.nn.functional as F

from megatron.core.inference.contexts.attention_context.dsa_metadata import (
    SCORE_ROW_BLOCK,
    DSARowPlan,
)
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext

from . import dsa_gqa_inference
from . import dsa_gqa_inference_kernels as kernels

try:
    import deep_select
except ImportError as error:
    raise ImportError(
        "Batch-invariant simplified DSA requires DeepSelect; install the batch_invariant "
        "dependency group."
    ) from error


def project(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Use the active BI projection backend identically for training and generation."""
    from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
        get_batch_invariant_backend,
        is_batch_invariant_mode_enabled,
        mm_batch_invariant,
    )

    rows = hidden.reshape(-1, hidden.size(-1))
    if is_batch_invariant_mode_enabled() and get_batch_invariant_backend() == "te_native":
        # The native BI runtime fixes the reduction through workspace starvation. Calling
        # mm_batch_invariant directly would instead substitute the much slower Triton GEMM.
        output = F.linear(rows, weight)
    else:
        output = mm_batch_invariant(rows, weight.T)
    return output.reshape(*hidden.shape[:-1], weight.size(0))


def aligned_score_buffer_width(max_seq_length: int, topk: int) -> int:
    """Align the model's maximum sequence length for DeepSelect score storage.
    DeepSelect internal tiling and reduction scheduling may change when per-request
    length varies. So we align this to the maximum sequence length.
    """
    if not 0 < max_seq_length <= int(2**20):  # 1M context
        raise ValueError(
            f"Batch-invariant DSA requires a model maximum sequence length in 1M, "
            f"got {max_seq_length}"
        )
    alignment, _ = deep_select.get_stride_requirement()
    elements = alignment // torch.float32.itemsize
    return -(-max(max_seq_length, topk) // elements) * elements


def select(
    scores: torch.Tensor, lengths: torch.Tensor, topk: int, *, out: torch.Tensor | None = None
) -> torch.Tensor:
    """Select request-local positions in ascending order
    (to support fixed order for tied scores), with trailing -1 padding."""
    _, indices = deep_select.topk(
        scores,
        topk,
        end=lengths.contiguous(),
        indices_type=torch.int32,
        sorted_index=True,
        # The pinned FP32 shortcut writes a sorted valid prefix followed by this sentinel;
        # long rows have no padding. No additional sorting or sentinel conversion is needed.
        idx_oob_fill_value=-1,
        output_idx=out,
        return_value=False,
    )
    return indices


@torch.no_grad()
def indexer_scores(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    block_table: torch.Tensor,
    row_lengths: torch.Tensor,
    block_starts: torch.Tensor,
    block_rows: torch.Tensor,
    block_requests: torch.Tensor,
    *,
    out: torch.Tensor,
    fill_invalid: bool = False,
    row_offset: int = 0,
    row_requests: torch.Tensor | None = None,
    workers: int | None = None,
) -> None:
    """Score each row's cached indexer keys on the SM100 TMA/TCGEN path.

    Takes the arguments of ``dsa_gqa_indexer_scores``. Strided indexer keys are copied to
    contiguous physical-page storage before constructing the descriptor.

    Args:
        fill_invalid: Write negative infinity at positions at or past each row's length.
            DeepSelect reads only ``[0, row_lengths[t])``, so selection leaves it off.
        row_requests: Optional direct row-to-request addressing for single-query blocks.
        workers: Optional score-grid parallelism override; every score retains the same MMA tile.
    """
    kernels._require_indexer_score_args(
        query,
        key_cache,
        block_table,
        row_lengths,
        block_starts,
        block_rows,
        block_requests,
        out,
        row_offset,
    )
    num_rows, index_dim = query.shape
    num_blocks = block_starts.numel()
    width = out.size(1)
    if num_blocks == 0 or width == 0:
        return
    if not kernels.HAVE_GLUON or index_dim != kernels.SCORE_INDEX_DIM:
        raise ValueError(
            f"Batch-invariant DSA scores require Blackwell Gluon and "
            f"index_dim={kernels.SCORE_INDEX_DIM}."
        )
    if key_cache.size(1) % kernels.SCORE_BLOCK_T and block_table.size(1) != 1:
        raise ValueError(
            f"Batch-invariant DSA requires {kernels.SCORE_BLOCK_T}-aligned KV pages or "
            f"single-page storage."
        )
    from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

    if fill_invalid:
        out.fill_(float("-inf"))
    shared_layout = kernels.gl.NVMMASharedLayout(
        swizzle_byte_width=128, element_bitwidth=16, rank=2, transposed=False
    )
    query_desc = TensorDescriptor.from_tensor(
        query.contiguous(), [kernels.BI_SCORE_ROWS, index_dim], shared_layout
    )
    key_desc = TensorDescriptor.from_tensor(
        key_cache.reshape(-1, index_dim).contiguous(),
        [kernels.SCORE_BLOCK_T, index_dim],
        shared_layout,
    )
    direct_rows = row_requests is not None and row_offset == 0 and out.size(0) == num_rows
    work_blocks = num_rows if direct_rows else num_blocks
    # Workers only assign independent 128-key score tiles to CTAs. Each score's MMA
    # and rounding remain identical when the grid changes with query scheduling.
    parallel_workers = max(1, kernels._num_sms(query.device.index) // work_blocks)
    workers = workers or min(
        128, -(-width // kernels.SCORE_BLOCK_T), max(8, 1 << (parallel_workers.bit_length() - 1))
    )
    kernels.score_tcgen[(work_blocks, workers)](
        query_desc,
        key_desc,
        block_table,
        row_lengths,
        row_requests if direct_rows else block_requests,
        out,
        stride_table=block_table.stride(0),
        WIDTH=width,
        PAGE=key_cache.size(1),
        WORKERS=workers,
        STAGES=3,
        BlockStarts=block_starts,
        BlockRows=block_rows,
        ROW_OFFSET=row_offset,
        OUT_ROWS=out.size(0),
        STRIDE_OUT=out.stride(0),
        SCORE_ROWS=kernels.BI_SCORE_ROWS,
        BATCH_INVARIANT=not direct_rows,
        SKIP_EMPTY_WORKERS=True,
        num_warps=4,
    )


@torch.no_grad()
def materialized_topk(
    *,
    index_query: torch.Tensor,
    index_key_cache: torch.Tensor,
    plan: DSARowPlan,
    score_buffer_width: int,
    index_topk: int,
) -> torch.Tensor:
    """Each row's top-k by scoring its keys into a fixed-width buffer and selecting from it."""

    def score_window(scores: torch.Tensor, start: int) -> None:
        indexer_scores(
            index_query,
            index_key_cache,
            plan.block_table,
            plan.row_lengths,
            plan.block_starts,
            plan.block_rows,
            plan.block_requests,
            out=scores,
            row_offset=start,
            row_requests=plan.row_requests if plan.decode_only else None,
        )

    def select_window(scores: torch.Tensor, start: int, stop: int, out: torch.Tensor) -> None:
        select(scores, plan.row_lengths[start:stop], index_topk, out=out)

    return dsa_gqa_inference._windowed_topk(
        num_rows=plan.row_requests.numel(),
        width=score_buffer_width,
        index_topk=index_topk,
        device=index_query.device,
        score_window=score_window,
        select_window=select_window,
    )


def _fixed_slices(topk: int, head_dim: int) -> int:
    """Partition of a row's selected keys that depends only on the head shape and top-k.

    A fixed support partition preserves every query's reduction tree across schedules.
    """
    return topk // kernels.BLOCK_N if topk == 2048 and head_dim == 128 else -(-topk // 256)


@torch.no_grad()
def sparse_attention_rows(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    index_query: torch.Tensor,
    index_key_cache: torch.Tensor,
    plan: DSARowPlan,
    score_buffer_width: int,
    index_topk: int,
    softmax_scale: float,
) -> torch.Tensor:
    """Select causal top-k keys per query token and attend over the paged KV cache.

    Prefill and decode rows share this path, so a token's output does not depend on the
    step it is computed in.
    """
    num_rows, num_query_heads, _ = query.shape
    block_table = plan.block_table.to(torch.int32).contiguous()
    plan = plan._replace(block_table=block_table)
    topk_indices = materialized_topk(
        index_query=index_query.contiguous(),
        index_key_cache=index_key_cache,
        plan=plan,
        score_buffer_width=score_buffer_width,
        index_topk=index_topk,
    )
    output = query.new_empty((num_rows, num_query_heads, value_cache.size(-1)))
    attention_args = (
        query,
        key_cache,
        value_cache,
        block_table,
        plan.row_requests,
        plan.row_lengths,
        topk_indices,
    )
    kernels._require_sparse_attention_args(*attention_args, output)
    kernels._launch_sparse_attention(
        *attention_args,
        softmax_scale,
        out=output,
        slices=_fixed_slices(index_topk, query.size(-1)),
        sorted_support=True,
        combine_kernel=kernels._combine_fixed_kernel,
    )
    return output


@torch.no_grad()
def dynamic_sparse_attention(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    index_query: torch.Tensor,
    index_key_cache: torch.Tensor,
    inference_context: DynamicInferenceContext,
    index_topk: int,
    softmax_scale: float,
    score_buffer_width: int,
) -> torch.Tensor:
    """Batch-invariant counterpart of ``dsa_gqa_inference.dynamic_sparse_attention``.

    Args:
        score_buffer_width: The indexer's fixed score-buffer width, shared with training.

    Returns:
        Attention output ``[T,1,Hq*Dv]``; rows for padding tokens are zero.
    """
    query, index_query, plan = dsa_gqa_inference._dynamic_rows(
        query, index_query, inference_context
    )
    if inference_context.max_sequence_length > score_buffer_width:
        raise ValueError("Inference context exceeds the model's BI DSA score-buffer width.")
    output = sparse_attention_rows(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        index_query=index_query,
        index_key_cache=index_key_cache,
        plan=plan,
        score_buffer_width=score_buffer_width,
        index_topk=index_topk,
        softmax_scale=softmax_scale,
    )
    return output.view(query.size(0), 1, -1)


def contiguous_plan(
    batch: int, length: int, start: int, stop: int, device: torch.device
) -> DSARowPlan:
    """View each contiguous training sequence as a single physical KV page."""
    count = stop - start
    if not 0 <= start < stop <= length:
        raise ValueError("Query interval must be a nonempty slice of the KV sequence.")
    requests = torch.arange(batch, dtype=torch.int32, device=device)
    positions = torch.arange(start + 1, stop + 1, dtype=torch.int32, device=device)
    blocks = torch.arange(0, count, SCORE_ROW_BLOCK, dtype=torch.int32, device=device)
    return DSARowPlan(
        requests.repeat_interleave(count),
        positions.repeat(batch),
        (requests[:, None] * count + blocks[None, :]).flatten(),
        (count - blocks).clamp_max(SCORE_ROW_BLOCK).repeat(batch),
        requests.repeat_interleave(blocks.numel()),
        torch.empty(0, dtype=torch.int32, device=device),
        requests[:, None],
        False,
    )


@torch.no_grad()
def route_contiguous(
    q_index: torch.Tensor,
    k_index: torch.Tensor,
    start: int,
    stop: int,
    topk: int,
    width: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the shared paged score and selection kernels for contiguous training inputs."""
    length, batch, _, dim = k_index.shape
    if length > width:
        raise ValueError("Training sequence exceeds the model's BI DSA score width.")
    plan = contiguous_plan(batch, length, start, stop, q_index.device)
    rows = q_index.permute(1, 0, 2, 3).reshape(-1, dim).contiguous()
    cache = k_index[:, :, 0].permute(1, 0, 2)
    indices = materialized_topk(
        index_query=rows,
        index_key_cache=cache,
        plan=plan,
        score_buffer_width=width,
        index_topk=topk,
    ).reshape(batch, stop - start, topk)
    # The auxiliary loss consumes scaled logits. Scaling is deliberately outside selection,
    # identically to inference, so BF16 projection rounding cannot move a top-k boundary.
    from .dsa_min_memory import _simplified_selected_index_scores

    scores = _simplified_selected_index_scores(q_index, k_index, indices, scale, start)
    return scores, indices


@torch.no_grad()
def attention_contiguous(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
    start: int,
) -> torch.Tensor:
    """Apply the shared forward to training SBHD tensors using stride-only KV views."""
    count, batch, heads, dim = query.shape
    plan = contiguous_plan(batch, key.size(0), start, start + count, query.device)
    rows = query.permute(1, 0, 2, 3).reshape(batch * count, heads, dim).contiguous()
    output = value.new_empty(batch * count, heads, value.size(-1))
    topk_indices = indices.reshape(batch * count, -1).to(torch.int32)
    attention_args = (
        rows,
        key.permute(1, 0, 2, 3),
        value.permute(1, 0, 2, 3),
        plan.block_table,
        plan.row_requests,
        plan.row_lengths,
        topk_indices,
    )
    kernels._require_sparse_attention_args(*attention_args, output)
    kernels._launch_sparse_attention(
        *attention_args,
        scale,
        out=output,
        slices=_fixed_slices(topk_indices.size(1), dim),
        sorted_support=True,
        combine_kernel=kernels._combine_fixed_kernel,
    )
    return output.reshape(batch, count, heads, value.size(-1)).permute(1, 0, 2, 3)
