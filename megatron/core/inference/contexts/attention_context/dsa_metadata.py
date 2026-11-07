# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Per-token metadata shared by DSA inference backends."""

from dataclasses import dataclass
from typing import NamedTuple, Sequence

import torch

from .metadata_base import MetadataBase

SCORE_ROW_BLOCK = 32


@dataclass(frozen=True)
class DSAPrefillSpan:
    """Query-row interval and causal KV length of one prefill request."""

    request: int
    query_start: int
    query_end: int
    kv_length: int


@dataclass(frozen=True)
class DSAStepInfo:
    """CPU request facts shared by DSA backends, independent of their GPU plans."""

    num_decode_requests: int
    max_decode_query_length: int
    max_decode_kv_length: int
    max_kv_length: int
    prefill_spans: tuple[DSAPrefillSpan, ...]


class DSARowPlan(NamedTuple):
    """GPU views for the Triton row kernels, sized only by the padded batch."""

    row_requests: torch.Tensor
    row_lengths: torch.Tensor
    block_starts: torch.Tensor
    block_rows: torch.Tensor
    block_requests: torch.Tensor
    decode_lengths: torch.Tensor
    block_table: torch.Tensor
    decode_only: bool


class DSAMetadata(MetadataBase):
    """Own DSA row metadata in pinned CPU and fixed-address GPU buffers.

    Row ``t`` belongs to request ``row_requests[t]`` and attends to its causal
    prefix ``[0, row_lengths[t])``. Consecutive rows of each request form score
    blocks; padding rows and blocks are empty. CPU updates are transferred with
    the context bookkeeping before eager execution or CUDA graph replay.

    Args:
        max_requests: Maximum padded request count.
        max_tokens: Maximum padded query-token count.
    """

    def __init__(self, *, max_requests: int, max_tokens: int) -> None:
        super().__init__()
        # A request with q query tokens takes ceil(q / block) <= q / block + 1 blocks.
        self._max_score_blocks = max_requests + -(-max_tokens // SCORE_ROW_BLOCK)
        sizes = [max_tokens, max_tokens, max_requests] + [self._max_score_blocks] * 3
        self._cpu_buffer = torch.zeros(sum(sizes), dtype=torch.int32, pin_memory=True)
        self._gpu_buffer = torch.zeros(
            sum(sizes), dtype=torch.int32, device=torch.cuda.current_device()
        )
        (
            self._cpu_row_requests,
            self._cpu_row_lengths,
            self._cpu_decode_lengths,
            self._cpu_block_starts,
            self._cpu_block_rows,
            self._cpu_block_requests,
        ) = self._cpu_buffer.split(sizes)
        (
            self.row_requests,
            self.row_lengths,
            self.decode_lengths,
            self.block_starts,
            self.block_rows,
            self.block_requests,
        ) = self._gpu_buffer.split(sizes)
        self.topk_workspace = None
        self.reset()

    def update(
        self,
        query_lengths: Sequence[int],
        kv_lengths: Sequence[int],
        *,
        num_decode_requests: int,
        padded_tokens: int,
        padded_requests: int,
        decode_only: bool,
    ) -> None:
        """Record CPU step facts and stage a row plan; decode requests precede prefill."""
        if len(query_lengths) != len(kv_lengths):
            raise ValueError("query_lengths and kv_lengths must have one entry per request.")
        if not 0 <= num_decode_requests <= len(query_lengths):
            raise ValueError("num_decode_requests must be within the real request count.")
        if any(length != 1 for length in query_lengths[:num_decode_requests]):
            raise NotImplementedError("DSA inference decodes one token per request.")

        spans = []
        query_start = 0
        for request, (query_length, kv_length) in enumerate(zip(query_lengths, kv_lengths)):
            query_end = query_start + query_length
            if request >= num_decode_requests:
                spans.append(DSAPrefillSpan(request, query_start, query_end, kv_length))
            query_start = query_end
        self._step = DSAStepInfo(
            num_decode_requests=num_decode_requests,
            max_decode_query_length=max(query_lengths[:num_decode_requests], default=0),
            max_decode_kv_length=max(kv_lengths[:num_decode_requests], default=0),
            max_kv_length=max(kv_lengths, default=0),
            prefill_spans=tuple(spans),
        )

        query_lengths_cpu = torch.tensor(query_lengths, dtype=torch.int64)
        kv_lengths_cpu = torch.tensor(kv_lengths, dtype=torch.int64)
        row_requests, row_lengths, block_starts, block_rows, block_requests = build_row_metadata(
            query_lengths_cpu, kv_lengths_cpu
        )
        real_tokens, num_blocks = row_requests.numel(), block_starts.numel()
        self.num_rows = padded_tokens
        self._decode_rows = min(padded_tokens, padded_requests)
        self._decode_only = decode_only
        self._cpu_row_requests[:real_tokens] = row_requests
        self._cpu_row_lengths[:real_tokens] = row_lengths
        self._cpu_row_requests[real_tokens:padded_tokens] = -1
        self._cpu_row_lengths[real_tokens:padded_tokens] = 0
        self._cpu_decode_lengths.zero_()
        self._cpu_decode_lengths[:num_decode_requests] = kv_lengths_cpu[:num_decode_requests]

        # Launch dimensions depend only on padding, so a captured graph covers
        # every batch that fits the same padded dimensions.
        self.num_score_blocks = padded_requests + -(-padded_tokens // SCORE_ROW_BLOCK)
        assert num_blocks <= self.num_score_blocks <= self._max_score_blocks
        self._cpu_block_starts[:num_blocks] = block_starts
        self._cpu_block_rows[:num_blocks] = block_rows
        self._cpu_block_requests[:num_blocks] = block_requests
        self._cpu_block_starts[num_blocks : self.num_score_blocks] = 0
        self._cpu_block_rows[num_blocks : self.num_score_blocks] = 0
        self._cpu_block_requests[num_blocks : self.num_score_blocks] = 0

    def transfer(self, block_table: torch.Tensor, non_blocking: bool) -> None:
        """Bind the native page table and copy the plan alongside context bookkeeping.

        Callers must wait before reusing an in-flight CPU staging buffer.
        """
        self._block_table = block_table
        self._gpu_buffer.copy_(self._cpu_buffer, non_blocking=non_blocking)

    def step(self) -> DSAStepInfo:
        """Return the host-side facts recorded by the latest update."""
        return self._step

    def plan(self) -> DSARowPlan:
        """Return the row backend's GPU plan for the padded step."""
        assert self._block_table is not None, "transfer() must bind the step's block table first."
        return DSARowPlan(
            row_requests=self.row_requests[: self.num_rows],
            row_lengths=self.row_lengths[: self.num_rows],
            block_starts=self.block_starts[: self.num_score_blocks],
            block_rows=self.block_rows[: self.num_score_blocks],
            block_requests=self.block_requests[: self.num_score_blocks],
            decode_lengths=self.decode_lengths[: self._decode_rows],
            block_table=self._block_table,
            decode_only=self._decode_only,
        )

    def reset(self) -> None:
        """Forget the step while preserving graph-owned metadata and shared top-k scratch."""
        self._step = DSAStepInfo(0, 0, 0, 0, ())
        self.num_rows = 0
        self.num_score_blocks = 0
        self._decode_rows = 0
        self._decode_only = True
        self._block_table = None


def build_row_metadata(
    query_lengths: torch.Tensor, kv_lengths: torch.Tensor, row_block: int = SCORE_ROW_BLOCK
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rows and score blocks for requests that each add ``query_lengths[r]`` query tokens.

    Request ``r``'s tokens are rows ``cu[r] .. cu[r+1]-1`` (``cu`` the cumulative query
    lengths) and the last of them sits at position ``kv_lengths[r] - 1``. Each request's rows
    are split into blocks of at most ``row_block`` rows.

    Args:
        query_lengths: Int CPU tensor ``[requests]`` of new query tokens per request.
        kv_lengths: Int CPU tensor ``[requests]`` of keys per request, the new ones included.
        row_block: Rows per score block.

    Returns:
        ``(row_requests, row_lengths, block_starts, block_rows, block_requests)``, int32 CPU
        tensors over ``sum(query_lengths)`` rows and the blocks.
    """
    query_lengths = query_lengths.to(torch.int64)
    kv_lengths = kv_lengths.to(torch.int64)
    num_requests = query_lengths.numel()
    num_rows = int(query_lengths.sum())
    request_ids = torch.arange(num_requests)
    first_row = torch.cumsum(query_lengths, 0) - query_lengths
    row_requests = torch.repeat_interleave(request_ids, query_lengths)
    within = torch.arange(num_rows) - first_row[row_requests]
    row_lengths = (kv_lengths - query_lengths)[row_requests] + within + 1
    blocks_per_request = (query_lengths + row_block - 1) // row_block
    block_requests = torch.repeat_interleave(request_ids, blocks_per_request)
    first_block = torch.cumsum(blocks_per_request, 0) - blocks_per_request
    block_in_request = torch.arange(int(blocks_per_request.sum())) - first_block[block_requests]
    block_starts = first_row[block_requests] + block_in_request * row_block
    block_rows = torch.clamp(
        query_lengths[block_requests] - block_in_request * row_block, max=row_block
    )
    return tuple(
        tensor.to(torch.int32)
        for tensor in (row_requests, row_lengths, block_starts, block_rows, block_requests)
    )
