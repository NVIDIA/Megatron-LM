# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CSA2 candidate blocks, with bounded scoring workspace and compact shared state.

Reuse DSv4's cuDNN BF16/MXFP8 scorer and radix Top-K. Only block reduction,
newest-block retention and deterministic compaction require new kernels.
Selection is discrete; indexer supervision keeps the original projection graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

from megatron.core.quantization.indexer_quantization import (
    create_indexer_mxfp8_quantization_buffers,
    indexer_mxfp8_thd_scale_capacity,
    indexer_mxfp8_thd_scale_shape,
    make_indexer_mxfp8_scale_cu_seqlens,
    quantize_indexer_mxfp8,
)

from . import fused_sparse_attention
from .thd_utils import CSA2THDCompressionLayout, CSA2THDLayout

if TYPE_CHECKING:
    from .csa2_indexer import CSA2IndexerInputs

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    triton = None
    tl = None
    HAVE_TRITON = False


# Score tiles are transient, including on the candidate-source layer. The
# unavoidable output is [queries, topk_blocks], never [queries, global keys].
_SCORE_CHUNK_MAX_BYTES = 32 * 1024 * 1024
_QUERY_CHUNK_SIZE = 128


@dataclass(frozen=True)
class CSA2CandidateBlocks:
    """Sorted sequence-local block IDs and valid counts, both int32.

    ``indices`` has shape [B, S, min(topk_blocks, num_blocks)] for SBHD or
    [T, min(topk_blocks, num_blocks)] for THD. Unavailable slots are -1 and
    follow valid IDs. Block zero starts at each sequence's first global key;
    physical THD padding never changes block alignment. ``lengths`` omits
    the final axis. No score tensor or dense position mask is retained.
    """

    indices: Tensor
    lengths: Tensor
    block_size: int

    def to_mask(
        self,
        num_keys: int,
        *,
        thd_layout: CSA2THDLayout | None = None,
        compressed_layout: CSA2THDCompressionLayout | None = None,
    ) -> Tensor:
        """Expand for the native reference and callers explicitly requesting scores.

        SBHD includes the full selected block; scoring still applies causality.
        THD returns a causal physical-row mask and excludes padding/other sequences.
        """
        shape = (*self.indices.shape[:-1], num_keys)
        if num_keys == 0 or self.indices.shape[-1] == 0:
            return torch.zeros(shape, dtype=torch.bool, device=self.indices.device)
        width = num_keys if thd_layout is None else compressed_layout.max_seqlen
        num_blocks = (width + self.block_size - 1) // self.block_size
        flags = torch.zeros(
            (*self.indices.shape[:-1], num_blocks + 1), dtype=torch.bool, device=self.indices.device
        )
        # An extra column receives invalid slots, so they cannot overwrite block 0.
        flags.scatter_(-1, self.indices.masked_fill(self.indices < 0, num_blocks).long(), True)
        if thd_layout is None:
            return flags[..., :num_blocks].repeat_interleave(self.block_size, -1)[..., :num_keys]
        local_blocks = compressed_layout.position_ids // compressed_layout.ratio // self.block_size
        selected = flags.gather(1, local_blocks.unsqueeze(0).expand(shape))
        return (
            selected
            & thd_layout.valid_tokens[:, None]
            & compressed_layout.valid_groups[None, :]
            & (thd_layout.sequence_ids[:, None] == compressed_layout.sequence_ids[None, :])
            & (
                compressed_layout.position_ids[None, :] + compressed_layout.ratio - 1
                <= thd_layout.position_ids[:, None]
            )
        )


@torch.no_grad()
def candidate_blocks_from_scores(
    scores: Tensor, visible: Tensor, topk_blocks: int, block_size: int
) -> CSA2CandidateBlocks:
    """Native block selection over already-masked, sequence-local score columns."""
    if topk_blocks <= 0 or block_size <= 0:
        raise ValueError("CSA2 candidate block count and size must be positive")
    width = scores.shape[-1]
    padded = F.pad(scores, (0, -width % block_size), value=-torch.inf)
    block_scores = padded.reshape(
        *scores.shape[:-1], padded.shape[-1] // block_size, block_size
    ).amax(-1)
    num_blocks = block_scores.shape[-1]
    newest = (visible - 1) // block_size
    block_scores = block_scores.masked_fill(
        torch.arange(num_blocks, device=scores.device) == newest.unsqueeze(-1), torch.inf
    )
    ids = block_scores.argsort(dim=-1, descending=True, stable=True)[
        ..., : min(topk_blocks, num_blocks)
    ]
    valid = block_scores.gather(-1, ids) > -torch.inf
    ids = ids.masked_fill(~valid, torch.iinfo(torch.int32).max).sort(-1).values
    return CSA2CandidateBlocks(
        ids.masked_fill(ids == torch.iinfo(torch.int32).max, -1).int(),
        valid.sum(-1).int(),
        block_size,
    )


if HAVE_TRITON:

    @triton.jit
    def _block_max_kernel(
        scores,
        visible,
        output,
        block_lengths,
        SCORE_STRIDE: tl.constexpr,
        NUM_KEYS: tl.constexpr,
        NUM_BLOCKS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        TILE_BLOCKS: tl.constexpr,
        TILE_KEYS: tl.constexpr,
    ):
        row = tl.program_id(0)
        block_ids = tl.program_id(1) * TILE_BLOCKS + tl.arange(0, TILE_BLOCKS)
        offsets = tl.arange(0, TILE_KEYS)
        keys = block_ids[:, None] * BLOCK_SIZE + offsets[None, :]
        length = tl.load(visible + row)
        values = tl.load(
            scores + row * SCORE_STRIDE + keys,
            (keys < tl.minimum(length, NUM_KEYS)) & (offsets[None, :] < BLOCK_SIZE),
            other=-float("inf"),
        )
        maxima = tl.max(values, 1)
        maxima = tl.where(
            (length > 0) & (block_ids == (length - 1) // BLOCK_SIZE), float("inf"), maxima
        )
        tl.store(output + row * NUM_BLOCKS + block_ids, maxima, block_ids < NUM_BLOCKS)
        if tl.program_id(1) == 0:
            tl.store(block_lengths + row, tl.cdiv(length, BLOCK_SIZE))

    @triton.jit
    def _compact_blocks_kernel(
        scores,
        top_indices,
        block_lengths,
        output,
        lengths,
        NUM_BLOCKS: tl.constexpr,
        TOPK: tl.constexpr,
        TILE_TOPK: tl.constexpr,
        TILE: tl.constexpr,
    ):
        """Keep exact radix winners, resolving boundary ties by the earlier block.

        Do not use ordinary indexer sanitize: +inf here deliberately retains the
        newest block. Prefix sums also put valid IDs first in ascending order.
        """
        row = tl.program_id(0)
        count = tl.load(block_lengths + row)
        out_cols = tl.arange(0, TILE_TOPK)
        tl.store(output + row * TOPK + out_cols, -1, out_cols < TOPK)
        ids = tl.load(top_indices + row * TOPK + out_cols, out_cols < TOPK, other=-1)
        values = tl.load(
            scores + row * NUM_BLOCKS + ids,
            (out_cols < TOPK) & (ids >= 0) & (ids < count),
            other=float("inf"),
        )
        threshold = tl.where(count <= TOPK, -float("inf"), tl.min(values, 0))
        offsets = tl.arange(0, TILE)
        num_greater = tl.full((), 0, tl.int32)
        for start in range(0, tl.cdiv(NUM_BLOCKS, TILE)):
            cols = start * TILE + offsets
            vals = tl.load(scores + row * NUM_BLOCKS + cols, cols < count, other=-float("inf"))
            num_greater += tl.sum(((cols < count) & (vals > threshold)).to(tl.int32), 0)
        tie_budget = TOPK - num_greater
        num_equal = tl.full((), 0, tl.int32)
        num_selected = tl.full((), 0, tl.int32)
        for start in range(0, tl.cdiv(NUM_BLOCKS, TILE)):
            cols = start * TILE + offsets
            vals = tl.load(scores + row * NUM_BLOCKS + cols, cols < count, other=-float("inf"))
            valid = (cols < count) & (vals > -float("inf"))
            equal = valid & (vals == threshold)
            tie_rank = num_equal + tl.cumsum(equal.to(tl.int32), 0)
            keep = valid & ((vals > threshold) | (equal & (tie_rank <= tie_budget)))
            ranks = num_selected + tl.cumsum(keep.to(tl.int32), 0) - 1
            tl.store(output + row * TOPK + ranks, cols, keep & (ranks < TOPK))
            num_selected += tl.sum(keep.to(tl.int32), 0)
            num_equal += tl.sum(equal.to(tl.int32), 0)
        tl.store(lengths + row, num_selected)


@torch.no_grad()
def _select_blocks(
    scores: Tensor, visible: Tensor, topk_blocks: int, block_size: int
) -> CSA2CandidateBlocks:
    """Reduce one score chunk and reuse cuDNN radix selection on its block maxima."""
    rows, width = scores.shape
    num_blocks = (width + block_size - 1) // block_size
    topk = min(topk_blocks, num_blocks)
    # The radix kernel uses 256-bit vector loads. Its row starts must be
    # 32-byte aligned; an odd stride can produce duplicate winners. Reduction
    # also fills the aligned tail with -inf, outside every row's valid length.
    score_stride = triton.cdiv(num_blocks, 8) * 8
    block_scores = torch.empty((rows, score_stride), device=scores.device, dtype=torch.float32)
    block_lengths = torch.empty((rows,), device=scores.device, dtype=torch.int32)
    _block_max_kernel[(rows, triton.cdiv(score_stride, 32))](
        scores,
        visible,
        block_scores,
        block_lengths,
        scores.stride(0),
        width,
        score_stride,
        block_size,
        32,
        triton.next_power_of_2(block_size),
    )
    indices, lengths = select_compact_positions(block_scores, block_lengths, topk)
    return CSA2CandidateBlocks(indices, lengths, block_size)


@torch.no_grad()
def select_compact_positions(
    scores: Tensor, valid_lengths: Tensor, topk: int
) -> tuple[Tensor, Tensor]:
    """Reuse radix Top-K and compact exact winners with stable boundary ties.

    FP32 score rows must be 32-byte aligned; masked positions contain -inf.
    Valid output positions are sorted, followed by -1. Candidate block scores
    may contain +inf to retain the newest block.
    """
    rows, score_stride = scores.shape
    if scores.stride(0) % 8 or scores.stride(1) != 1 or not 0 < topk <= score_stride:
        raise ValueError("Radix scores require aligned rows and 0 < topk <= row width")
    fused_sparse_attention._ensure_dsa_namespace()
    result = fused_sparse_attention._DSA.indexer_top_k_wrapper(
        scores, valid_lengths, top_k=topk, next_n=1, return_val=False
    )
    indices = torch.empty((rows, topk), dtype=torch.int32, device=scores.device)
    lengths = torch.empty_like(valid_lengths)
    _compact_blocks_kernel[(rows,)](
        scores,
        result["indices"],
        valid_lengths,
        indices,
        lengths,
        score_stride,
        topk,
        triton.next_power_of_2(topk),
        1024,
    )
    return indices, lengths


@torch.no_grad()
def fused_candidate_blocks(
    inputs: CSA2IndexerInputs,
    *,
    topk_blocks: int,
    block_size: int,
    precision: str,
    token_topk: int | None = None,
) -> CSA2CandidateBlocks | tuple[Tensor, CSA2CandidateBlocks]:
    """Generate candidates using chunked DSv4 scores and compact block selection.

    Consume the same prepared Q/K/W as this layer's indexer loss. SBHD batches
    are viewed as packed segments so workspace does not grow with batch size.
    Each chunk preserves the original sequence's causal offset.
    K is quantized once in MXFP8 mode; each Q chunk is quantized independently
    with the same channel-block rounding. No round trip to dense FP32 Q/K is used.
    When ``token_topk`` is supplied, select this layer's token Top-K from the
    same score tiles before releasing them. Candidates never restrict that Top-K.
    """
    q, k, weights = inputs.q, inputs.k, inputs.weights
    ratio, thd_layout = inputs.ratio, inputs.thd_layout
    if ratio not in (1, 2) or topk_blocks <= 0 or block_size <= 0:
        raise ValueError("CSA2 candidates require ratio 1/2 and positive block count/size")
    if precision not in ("bf16", "mxfp8"):
        raise ValueError("CSA2 candidate precision must be bf16 or mxfp8")
    if not HAVE_TRITON:
        raise RuntimeError("Fused CSA2 candidates require Triton")
    if not q.is_cuda or any(
        t.dtype != torch.bfloat16 or t.device != q.device for t in (q, k, weights)
    ):
        raise ValueError("Fused CSA2 candidate projections must be CUDA BF16 tensors")
    if q.shape[-2] not in (32, 64) or q.shape[-1] != 128 or k.shape[-1] != 128:
        raise ValueError("Fused CSA2 candidates require H32/H64 and D128")
    num_heads = q.shape[-2]
    token_width = min(token_topk, inputs.key_capacity) if token_topk is not None else 0
    # Preserve selection rounding without modifying the shared, unscaled W.
    scaled_weights = (weights.float() * (128**-0.5 * num_heads**-0.5)).to(weights.dtype)
    output_shape, max_k, visible = inputs.output_shape, inputs.max_keys, inputs.visible
    num_blocks = (max_k + block_size - 1) // block_size
    width = min(topk_blocks, num_blocks)
    indices = torch.full((q.shape[0], width), -1, dtype=torch.int32, device=q.device)
    lengths = torch.zeros(q.shape[0], dtype=torch.int32, device=q.device)
    token_indices = torch.full((q.shape[0], token_width), -1, dtype=torch.int32, device=q.device)

    def result():
        blocks = CSA2CandidateBlocks(
            indices.reshape(*output_shape, width), lengths.reshape(output_shape), block_size
        )
        if token_topk is None:
            return blocks
        if thd_layout is not None:
            starts = inputs.starts.unsqueeze(-1)
            selected = torch.where(token_indices >= 0, token_indices + starts, -1)
        else:
            selected = token_indices
        return selected.reshape(*output_shape, token_width), blocks

    if q.shape[0] == 0 or k.shape[0] == 0 or max_k == 0:
        return result()

    fused_sparse_attention._ensure_dsa_namespace()
    cu_q, cu_k, max_q, score_max_k = inputs.packed_metadata
    # Account for cuDNN's four-float row alignment. Even a single very long
    # row needs its own score buffer; otherwise cap the chunk at 32 MiB.
    row_bytes = ((score_max_k + 3) // 4 * 4) * 4
    chunk_size = min(_QUERY_CHUNK_SIZE, max(1, _SCORE_CHUNK_MAX_BYTES // row_bytes))
    kernel_k = k
    precision_kwargs = dict(precision=precision)
    if precision == "mxfp8":
        # Reuse DSv4's static scale bounds: reading a device prefix to size
        # these buffers would synchronize once for K and once per Q chunk.
        num_sequences = cu_q.numel() - 1
        chunk_rows = min(chunk_size, q.shape[0])
        q_scale_capacity = indexer_mxfp8_thd_scale_capacity(chunk_rows, num_sequences, 64)
        k_scale_capacity = indexer_mxfp8_thd_scale_capacity(k.shape[0], num_sequences, 1)
        q_scale = torch.empty(
            indexer_mxfp8_thd_scale_shape(q_scale_capacity, 64, q.shape[-1]),
            dtype=torch.float8_e8m0fnu,
            device=q.device,
        )
        k_scale = torch.empty(
            indexer_mxfp8_thd_scale_shape(k_scale_capacity, 1, k.shape[-1]),
            dtype=torch.float8_e8m0fnu,
            device=k.device,
        )
        k_scale_prefix = make_indexer_mxfp8_scale_cu_seqlens(cu_k, 1)
        kernel_k, k_scale = quantize_indexer_mxfp8(
            k, cu_seqlens=cu_k, cu_seqlens_scale_padded=k_scale_prefix, out_scale=k_scale
        )
        precision_kwargs.update(k_scale=k_scale, cu_seqlens_k_scale_padded=k_scale_prefix)
        q_buffers = None
        if num_heads == 32:
            padded_q = q.new_zeros((chunk_rows, 64, q.shape[-1]))
            padded_w = weights.new_zeros((chunk_rows, 64))

    for start in range(0, q.shape[0], chunk_size):
        end = min(start + chunk_size, q.shape[0])
        chunk_cu_q = (cu_q.clamp(min=start, max=end) - start).contiguous()
        offsets = (start - cu_q[:-1]).clamp_min(0).contiguous()
        kernel_q = q[start:end]
        kernel_w = scaled_weights[start:end]
        if precision == "mxfp8":
            # The existing MXFP8 scorer needs H64. Pad only this chunk; keep
            # real H32 scaling and avoid doubling the full projected Q storage.
            if num_heads == 32:
                padded_q[: end - start, :num_heads].copy_(kernel_q)
                padded_w[: end - start, :num_heads].copy_(kernel_w)
                kernel_q, kernel_w = padded_q[: end - start], padded_w[: end - start]
            # Only a shorter final chunk needs a different TE destination.
            # Scale storage also covers its changing packed-sequence boundaries.
            if q_buffers is None or not q_buffers.matches(kernel_q):
                q_buffers = create_indexer_mxfp8_quantization_buffers(kernel_q)
            q_scale_prefix = make_indexer_mxfp8_scale_cu_seqlens(chunk_cu_q, kernel_q.shape[-2])
            kernel_q, q_scale = quantize_indexer_mxfp8(
                kernel_q,
                cu_seqlens=chunk_cu_q,
                cu_seqlens_scale_padded=q_scale_prefix,
                buffers=q_buffers,
                out_scale=q_scale,
            )
            precision_kwargs.update(q_scale=q_scale, cu_seqlens_q_scale_padded=q_scale_prefix)
        scores = fused_sparse_attention._DSA.indexer_forward_wrapper(
            kernel_q,
            kernel_k.unsqueeze(1),
            kernel_w,
            ratio=ratio,
            cu_seqlens_q=chunk_cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=min(end - start, max_q),
            max_seqlen_k=score_max_k,
            q_causal_offsets=offsets,
            **precision_kwargs,
        )["scores"]
        if token_width:
            # cuDNN's scorer aligns to four floats; radix needs eight. Only
            # this bounded tile is repacked when the two strides differ.
            token_scores = scores
            aligned = triton.cdiv(max(scores.shape[1], token_width), 8) * 8
            if scores.stride(0) % 8 or aligned > scores.shape[1]:
                token_scores = F.pad(scores, (0, aligned - scores.shape[1]), value=-torch.inf)
            selected_tokens, _ = select_compact_positions(
                token_scores, visible[start:end], token_width
            )
            token_indices[start:end].copy_(selected_tokens)
            del token_scores, selected_tokens
        selected = _select_blocks(scores[:, :max_k], visible[start:end], topk_blocks, block_size)
        indices[start:end].copy_(selected.indices)
        lengths[start:end].copy_(selected.lengths)
        # Release before the next scorer allocates its chunk (no two full score tiles).
        del scores, selected
    return result()
