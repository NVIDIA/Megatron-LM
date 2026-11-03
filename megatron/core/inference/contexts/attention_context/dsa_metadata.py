# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Per-token metadata shared by DSA inference backends."""

import torch

SCORE_ROW_BLOCK = 32


class DSAMetadata:
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
        self.num_rows = 0
        self.num_score_blocks = 0
        self.decode_workspaces = {}

    def update(
        self,
        query_lengths: torch.Tensor,
        kv_lengths: torch.Tensor,
        *,
        padded_request_count: int,
        padded_token_count: int,
    ) -> None:
        """Stage real requests' CPU lengths and clear padding for the next step."""
        row_requests, row_lengths, block_starts, block_rows, block_requests = build_row_metadata(
            query_lengths, kv_lengths
        )
        real_tokens, num_blocks = row_requests.numel(), block_starts.numel()
        self.num_rows = padded_token_count
        self._cpu_row_requests[:real_tokens] = row_requests
        self._cpu_row_lengths[:real_tokens] = row_lengths
        self._cpu_row_requests[real_tokens:padded_token_count] = -1
        self._cpu_row_lengths[real_tokens:padded_token_count] = 0
        self._cpu_decode_lengths.zero_()
        leading_decode = 0
        for length in query_lengths.tolist():
            if length != 1:
                break
            leading_decode += 1
        self._cpu_decode_lengths[:leading_decode] = kv_lengths[:leading_decode]

        # Launch dimensions depend only on padding, so a captured graph covers
        # every batch that fits the same padded dimensions.
        self.num_score_blocks = padded_request_count + -(-padded_token_count // SCORE_ROW_BLOCK)
        assert num_blocks <= self.num_score_blocks <= self._max_score_blocks
        self._cpu_block_starts[:num_blocks] = block_starts
        self._cpu_block_rows[:num_blocks] = block_rows
        self._cpu_block_requests[:num_blocks] = block_requests
        self._cpu_block_starts[num_blocks : self.num_score_blocks] = 0
        self._cpu_block_rows[num_blocks : self.num_score_blocks] = 0
        self._cpu_block_requests[num_blocks : self.num_score_blocks] = 0

    def transfer_to_gpu(self, *, non_blocking: bool = False) -> None:
        """Copy staged metadata; callers must wait before reusing an in-flight CPU buffer."""
        self._gpu_buffer.copy_(self._cpu_buffer, non_blocking=non_blocking)

    def row_metadata(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Return GPU row/request and score-block views for this step's padded dimensions."""
        return (
            self.row_requests[: self.num_rows],
            self.row_lengths[: self.num_rows],
            self.block_starts[: self.num_score_blocks],
            self.block_rows[: self.num_score_blocks],
            self.block_requests[: self.num_score_blocks],
            self.num_score_blocks,
        )


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
