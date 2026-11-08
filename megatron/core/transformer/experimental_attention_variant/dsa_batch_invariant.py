# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared routing and contiguous-storage adapters for batch-invariant simplified DSA.
Don't import this file if batch-invariant mode is not supported as it will error out
that deep_select is not installed.
"""

import torch
import torch.nn.functional as F

from megatron.core.inference.contexts.attention_context.dsa_metadata import (
    SCORE_ROW_BLOCK,
    DSARowPlan,
)

try:
    import deep_select
except ImportError as error:
    raise ImportError("Batch-invariant simplified DSA requires the inference extra.") from error


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


def score_width(max_seq_length: int | None, topk: int) -> int:
    """Align the model's maximum sequence length for DeepSelect score storage."""
    if max_seq_length is None or not 0 < max_seq_length <= int(2**20):  # 1M context
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
    """Select request-local positions in ascending order, with trailing -1 padding."""
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
    from .dsa_gqa_inference import _materialized_topk

    length, batch, _, dim = k_index.shape
    if length > width:
        raise ValueError("Training sequence exceeds the model's BI DSA score width.")
    plan = contiguous_plan(batch, length, start, stop, q_index.device)
    rows = q_index.permute(1, 0, 2, 3).reshape(-1, dim).contiguous()
    cache = k_index[:, :, 0].permute(1, 0, 2)
    indices = _materialized_topk(
        index_query=rows,
        index_key_cache=cache,
        plan=plan,
        max_context_length=width,
        index_topk=topk,
        batch_invariant=True,
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
    from .dsa_gqa_inference_kernels import dsa_gqa_sparse_attention

    count, batch, heads, dim = query.shape
    plan = contiguous_plan(batch, key.size(0), start, start + count, query.device)
    rows = query.permute(1, 0, 2, 3).reshape(batch * count, heads, dim).contiguous()
    output = value.new_empty(batch * count, heads, value.size(-1))
    dsa_gqa_sparse_attention(
        rows,
        key.permute(1, 0, 2, 3),
        value.permute(1, 0, 2, 3),
        plan.block_table,
        plan.row_requests,
        plan.row_lengths,
        indices.reshape(batch * count, -1).to(torch.int32),
        scale,
        out=output,
        batch_invariant=True,
    )
    return output.reshape(batch, count, heads, value.size(-1)).permute(1, 0, 2, 3)
