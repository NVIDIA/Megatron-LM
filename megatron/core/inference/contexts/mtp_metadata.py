# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import torch
from torch import Tensor

from megatron.core.inference.utils import tensor_swap

from .gpu_view import ContextGPUView


class MTPForwardMode(Enum):
    """Which MTP forward, if any, currently owns the attention metadata.

    The two non-NONE modes differ in query shape, which is what decides the attention kernel:
    a draft depth contributes exactly one token per active request (uniform, so the decode
    kernel's `q.reshape(num_requests, tokens_per_request, ...)` is valid), while the commit
    pass is roll-by-one over each request's committed span (ragged, so it must go varlen).
    """

    NONE = "none"
    DRAFT = "draft"
    COMMIT = "commit"


@dataclass
class MTPMetadata:
    """MTP request annotations, chunk carry, and KV-cache forward buffers.

    A speculative step runs several MTP forwards back to back -- one per draft depth, plus a
    varlen "commit pass" -- and every one of them rewrites the same small set of per-request
    fields. Rather than build those tensors per forward, this class owns them as fixed-address
    buffers sized for the worst case and updates them in place. That keeps the draft loop
    allocation-free and CUDA-graph safe (a captured MTP graph replays against the same
    addresses each step). Construction is cheap and unconditional; `allocate` is what reserves
    GPU memory, and only when `enabled`. CPU request annotations follow the context's
    row lifecycle through `move_request_rows`, `swap_request_rows`, and `reset_request_rows`.

    Args:
        enabled (bool): Whether MTP KV caching is active for this context. When False the
            object is inert: no buffers are allocated and `forward_mode` stays NONE.
        max_requests (int): Worst-case active request count (`max_bs`).
        max_kv_block_count (int): Block-table width, in blocks per request.
        block_size_tokens (int): KV block size, used to split a write position into
            (block column, offset within block).
        dummy_block_idx (int): Scratch block that padded rows read and write, so padding never
            touches real KV.
        block_table_dtype (torch.dtype): dtype of `gpu_view.mha_block_table`.
        hidden_size (int): Model hidden size, sizing the chunked-prefill boundary carry.
        hidden_dtype (torch.dtype): dtype of the main model's hidden states.
    """

    enabled: bool
    max_requests: int
    max_kv_block_count: int
    block_size_tokens: int
    dummy_block_idx: int
    block_table_dtype: torch.dtype
    hidden_size: int
    hidden_dtype: torch.dtype

    # CPU row metadata, indexed by the context's request rows (including paused rows).
    # Mirrors DynamicInferenceRequest.num_matched_prefix_blocks so the commit pass can
    # protect inherited KV without holding request objects. Allocated with the GPU buffers.
    request_matched_prefix_blocks: Optional[Tensor] = field(default=None, repr=False)

    # ---- Draft-loop state (valid between begin_decode() and end_decode()). ----
    # Which MTP forward owns the attention metadata right now. The KV append/read paths key off
    # this being non-NONE to route to the MTP layer slot instead of a main attention layer, and
    # `is_decode_only()` keys off COMMIT to force the varlen kernel.
    forward_mode: MTPForwardMode = MTPForwardMode.NONE
    # Whether this draft loop replays captured CUDA graphs (True) or runs eager (False).
    graphed: bool = False
    active_request_count: int = 0
    padded_count: int = 0
    # ---- Chunked-prefill boundary carry (lives BETWEEN steps). ----
    # The MTP entry at position `off - 1` straddles two prefill chunks: it pairs the previous
    # chunk's last hidden with the next chunk's first token, so neither chunk can write it alone.
    # The producing chunk stashes its last hidden here; the consuming chunk splices it in. See
    # `_mtp_prefill_commit_segments`.
    #
    # The carry records the producing request's id AND the prompt position the hidden was computed
    # at, and `take_chunk_boundary` asserts on both. The id alone is not sufficient -- see
    # that method for why.
    chunk_boundary_valid: bool = False
    chunk_boundary_req_id: int = -1
    chunk_boundary_position: int = -1
    # [1, 1, hidden_size]: preallocated and written in place, like every other buffer here.
    chunk_boundary_hidden: Optional[Tensor] = field(default=None, repr=False)

    # ---- Persistent buffers (allocated by allocate()). ----
    # [max_requests] int32: MTP write position of the current draft depth, per request.
    offsets: Optional[Tensor] = field(default=None, repr=False)
    # [max_requests, max_kv_block_count]: GPU copy of the block table the draft loop uses.
    block_table: Optional[Tensor] = field(default=None, repr=False)
    # [max_requests] int32: staging for the per-request MHA query/KV lengths.
    query_lengths: Optional[Tensor] = field(default=None, repr=False)
    kv_lengths: Optional[Tensor] = field(default=None, repr=False)
    # [max_requests] int64: constant [0, 1, ..., max_requests - 1].
    row_ids: Optional[Tensor] = field(default=None, repr=False)

    # ---- Views into the buffers, refreshed once per draft loop by begin_decode(). ----
    active_offsets: Optional[Tensor] = field(default=None, repr=False)
    active_block_table: Optional[Tensor] = field(default=None, repr=False)

    @property
    def forward_active(self) -> bool:
        """Whether any MTP forward currently owns the attention metadata."""
        return self.forward_mode is not MTPForwardMode.NONE

    @property
    def is_varlen_forward(self) -> bool:
        """Whether the current MTP forward has ragged per-request query lengths."""
        return self.forward_mode is MTPForwardMode.COMMIT

    def allocate(self, device: torch.device) -> None:
        """Reserve the persistent buffers. No-op when MTP KV caching is disabled.

        Args:
            device (torch.device): Device for the GPU-resident buffers.
        """
        if not self.enabled:
            return
        self.request_matched_prefix_blocks = torch.zeros(
            self.max_requests, dtype=torch.int32, device="cpu", pin_memory=True
        )
        self.offsets = torch.zeros(self.max_requests, dtype=torch.int32, device=device)
        self.block_table = torch.full(
            (self.max_requests, self.max_kv_block_count),
            self.dummy_block_idx,
            dtype=self.block_table_dtype,
            device=device,
        )
        self.query_lengths = torch.zeros(self.max_requests, dtype=torch.int32, device=device)
        self.kv_lengths = torch.zeros(self.max_requests, dtype=torch.int32, device=device)
        self.row_ids = torch.arange(self.max_requests, dtype=torch.int64, device=device)
        self.chunk_boundary_hidden = torch.zeros(
            (1, 1, self.hidden_size), dtype=self.hidden_dtype, device=device
        )
        self.invalidate_chunk_boundary()

    def deallocate(self) -> None:
        """Release the persistent buffers, mirroring `allocate`.

        Used by the context's suspend path, which drops its tensors and rebuilds them from
        `initialize_all_tensors` on resume.
        """
        self.request_matched_prefix_blocks = None
        self.offsets = None
        self.block_table = None
        self.query_lengths = None
        self.kv_lengths = None
        self.row_ids = None
        self.active_offsets = None
        self.active_block_table = None
        self.forward_mode = MTPForwardMode.NONE
        self.chunk_boundary_hidden = None
        self.invalidate_chunk_boundary()

    def reset_request_rows(self, request_indexes=slice(None)) -> None:
        """Clear MTP annotations when request rows are released, reused, or reset.

        The chunk carry is keyed by request ID, so moving or clearing physical rows
        does not change it. Its owner invalidates it when that logical request ends.
        """
        if self.enabled:
            self.request_matched_prefix_blocks[request_indexes] = 0

    def move_request_rows(self, src_idxs: Tensor, dst_idxs: Tensor) -> None:
        """Copy MTP annotations alongside the scheduler's request-row movement.

        Sources remain intact until the scheduler identifies and clears vacated rows.
        Advanced indexing snapshots the source values even when the ranges overlap.
        """
        if self.enabled:
            self.request_matched_prefix_blocks[dst_idxs] = self.request_matched_prefix_blocks[
                src_idxs
            ]

    def swap_request_rows(self, src_idxs: Tensor, dst_idxs: Tensor) -> None:
        """Swap MTP annotations alongside the scheduler's paused/active row swap."""
        if self.enabled:
            tensor_swap(self.request_matched_prefix_blocks, src_idxs, dst_idxs)

    # ------------------------------------------------------------------
    # Chunked-prefill boundary carry.
    # ------------------------------------------------------------------
    def invalidate_chunk_boundary(self) -> None:
        """Drop the carried chunk-boundary hidden.

        Clears the validity and the keys, not the buffer -- its address stays stable. Called from
        `deallocate` and whenever no chunked request is in flight, so a stale carry can never be
        matched by a later request.
        """
        self.chunk_boundary_valid = False
        self.chunk_boundary_req_id = -1
        self.chunk_boundary_position = -1

    def carry_chunk_boundary(self, hidden: Tensor, req_id: int, position: int) -> None:
        """Stash the in-flight chunked request's last hidden for its next chunk's seam.

        Args:
            hidden (Tensor): Main hidden state at `position`, any shape with `hidden_size`
                elements. Copied into the persistent buffer, so the caller's tensor is free to
                be released with the rest of the step's activations.
            req_id (int): Producing request's id.
            position (int): Prompt position `hidden` was computed at. Only a chunk whose seam
                falls exactly here -- i.e. whose `off == position + 1` -- may consume this carry.
        """
        if not self.enabled:
            return
        self.chunk_boundary_hidden.copy_(hidden.detach().view(1, 1, -1))
        self.chunk_boundary_req_id = req_id
        self.chunk_boundary_position = position
        self.chunk_boundary_valid = True

    def take_chunk_boundary(self, req_id: int, seam_position: int) -> Tensor:
        """Return the carried hidden for this seam.

        All three conditions below hold by construction. The caller derives `req_id` by matching
        `chunk_boundary_req_id` in its own request list, which settles the first two. The third
        rests on `_compute_prefix_match` giving up a carry-holding continuation chunk's ENTIRE
        prefix match, so the chunk starts at `finished` and its seam lands exactly where the
        carry was recorded.

        They RAISE rather than declining: skipping the seam would leave a committed position
        unwritten, a silent draft-acceptance regression that masks whichever invariant broke.

        Args:
            req_id (int): Request that wants to write the seam.
            seam_position (int): MTP position the seam would be written at (`off - 1`).

        Returns:
            (Tensor): The `[1, 1, hidden_size]` carried hidden.
        """
        assert self.chunk_boundary_valid, (
            f"no live chunk-boundary carry, but request {req_id} asked for one at position "
            f"{seam_position}. Callers must match `chunk_boundary_req_id` in their own request "
            f"list first, and an invalid carry records -1, which is never a real request id."
        )
        assert self.chunk_boundary_req_id == req_id, (
            f"chunk-boundary carry belongs to request {self.chunk_boundary_req_id}, not "
            f"{req_id}; the caller passed a request it did not match on."
        )
        assert self.chunk_boundary_position == seam_position, (
            f"chunk-boundary carry for request {req_id} sits at position "
            f"{self.chunk_boundary_position}, but its seam is at {seam_position}. A carry-holding "
            f"continuation chunk must take NO prefix skip -- check the back-off in "
            f"`_compute_prefix_match`."
        )
        return self.chunk_boundary_hidden

    # ------------------------------------------------------------------
    # Draft-loop lifecycle.
    # ------------------------------------------------------------------
    def begin_decode(
        self,
        active_request_count: int,
        padded_count: int,
        start_positions: Tensor,
        block_table_src: Tensor,
        graphed: bool,
    ) -> None:
        """Stage the draft loop's write positions and block table.

        Both are copied into the persistent buffers (rather than aliased), so advancing the
        depth cannot mutate the caller's tensors.

        Args:
            active_request_count (int): Requests taking part in the draft loop.
            padded_count (int): Request/token slots the forwards launch over.
            start_positions (Tensor): [>=active_request_count] depth-0 write position per request.
            block_table_src (Tensor): [>=active_request_count, max_kv_block_count] source table.
            graphed (bool): Whether the loop replays captured CUDA graphs.
        """
        n = active_request_count
        self.offsets[:n].copy_(start_positions[:n], non_blocking=True)
        self.block_table[:n].copy_(block_table_src[:n], non_blocking=True)
        self._enter_decode(n, padded_count, graphed)

    def begin_decode_for_capture(self, padded_count: int) -> None:
        """Stage synthetic (scratch-only) draft state for CUDA-graph capture at warmup.

        Every row is pointed at the dummy block at position 0, so the captured append/attend
        touch only scratch KV. Replay overwrites all of it, so only the shapes and the fixed
        launch bounds matter here, and those match the runtime graphed step.
        """
        assert self.enabled
        self.offsets[:padded_count].zero_()
        self.block_table[:padded_count].fill_(self.dummy_block_idx)
        self._enter_decode(padded_count, padded_count, graphed=True)

    def _enter_decode(self, active_request_count: int, padded_count: int, graphed: bool) -> None:
        """Enter MTP-forward mode and refresh the active views."""
        self.active_request_count = active_request_count
        self.padded_count = padded_count
        self.active_offsets = self.offsets[:active_request_count]
        self.active_block_table = self.block_table[:active_request_count]
        self.graphed = graphed
        self.forward_mode = MTPForwardMode.DRAFT

    def advance_decode_step(self) -> None:
        """Advance every active request's MTP write position by one, in place."""
        self.active_offsets += 1

    def end_forward(self) -> None:
        """Leave MTP-forward mode. No persistent length state to write back."""
        self.forward_mode = MTPForwardMode.NONE

    # ------------------------------------------------------------------
    # Per-forward metadata staging.
    # ------------------------------------------------------------------
    def stage_decode_lengths(self) -> Tuple[Tensor, Tensor]:
        """Stage the MHA lengths for one draft depth: one query row, `position + 1` keys.

        Returns:
            (Tuple[Tensor, Tensor]): Views of the query- and KV-length buffers, both
            `active_request_count` long.
        """
        n = self.active_request_count
        query_lengths = self.query_lengths[:n]
        query_lengths.fill_(1)
        kv_lengths = self.kv_lengths[:n]
        torch.add(self.active_offsets, 1, out=kv_lengths)
        return query_lengths, kv_lengths

    def stage_prefill_lengths(self, append_counts: Tensor, pad_tokens: int) -> Tensor:
        """Stage the MHA lengths for a varlen commit pass, absorbing the padded query rows.

        The commit pass is a fresh causal prefill, so a request's KV length equals its query
        length and one buffer serves as both.

        The hidden is padded to a TP multiple for the sequence-parallel scatter, so attention
        sees `total + pad_tokens` query rows while `append_counts` describes only `total`. Varlen
        requires `q.shape[0] == cu_seqlens_q[-1]`, so every pad row needs an owning request --
        and it does not matter which, since `write_token_maps` sends them to the dummy block.

        Args:
            append_counts (Tensor): [P] KV entries each request writes this pass.
            pad_tokens (int): Query rows appended to reach the TP multiple.

        Returns:
            (Tensor): Per-request length view, `P` long, or `P + 1` when the pad rows were given
            their own trailing request.
        """
        num_prefill = append_counts.numel()
        # A spare request slot carries the pad rows in their own trailing request, which attends
        # solely to itself out of the dummy block. A full batch occupies every slot, so extend
        # the last real request's run instead: its reads stay inside blocks it already owns,
        # since this varlen forward reads from the head of each block-table row and the run
        # grows by at most `tp_size - 1`.
        use_spare_slot = pad_tokens > 0 and num_prefill < self.max_requests
        num_requests = num_prefill + 1 if use_spare_slot else num_prefill
        lengths = self.query_lengths[:num_requests]
        lengths[:num_prefill].copy_(append_counts)
        if use_spare_slot:
            lengths[num_prefill:num_requests].fill_(pad_tokens)
        elif pad_tokens > 0:
            lengths[num_prefill - 1 : num_prefill] += pad_tokens
        return lengths

    # ------------------------------------------------------------------
    # GPU-view writes. Shared by every MTP forward: the decode and prefill setups differ only
    # in how they derive the (row, position) pairs and the per-request lengths passed in here.
    # ------------------------------------------------------------------
    def write_token_maps(
        self,
        gpu_view: ContextGPUView,
        rows: Tensor,
        positions: Tensor,
        block_table: Tensor,
        padded_token_count: int,
        inherited_blocks: Optional[Tensor] = None,
    ) -> None:
        """Write the per-token KV destination maps for one MTP forward.

        Args:
            gpu_view (ContextGPUView): Destination bookkeeping views.
            rows (Tensor): [T] owning request row (into `block_table`) for each token.
            positions (Tensor): [T] MTP write position within the request for each token.
            block_table (Tensor): [R, max_kv_block_count] block ids indexed by `rows`.
            padded_token_count (int): Token rows the forward runs, including padding. Padded
                rows are redirected to the dummy block so they never touch real KV.
            inherited_blocks (Optional[Tensor]): [R] leading blocks each row INHERITED rather
                than computed. Tokens landing in those blocks go to the dummy block: the KV is
                already correct from the producer, and the block is ref-counted, so writing would
                corrupt every request sharing it. None disables the redirect.
        """
        total = positions.numel()
        block_within = (positions // self.block_size_tokens).to(torch.long)

        destinations = block_table[rows, block_within]
        if inherited_blocks is not None:
            destinations = torch.where(
                block_within < inherited_blocks[rows],
                torch.full_like(destinations, self.dummy_block_idx),
                destinations,
            )
        gpu_view.token_to_block_idx[:total] = destinations.to(gpu_view.token_to_block_idx.dtype)
        gpu_view.token_to_local_position_within_kv_block[:total] = (
            positions % self.block_size_tokens
        ).to(gpu_view.token_to_local_position_within_kv_block.dtype)
        gpu_view.token_to_request_idx[:total] = rows.to(gpu_view.token_to_request_idx.dtype)
        gpu_view.token_to_position_in_request[:total] = positions.to(
            gpu_view.token_to_position_in_request.dtype
        )
        gpu_view.token_to_pos_ids[:total] = positions.to(gpu_view.token_to_pos_ids.dtype)

        if padded_token_count > total:
            gpu_view.token_to_block_idx[total:padded_token_count] = self.dummy_block_idx
            gpu_view.token_to_local_position_within_kv_block[total:padded_token_count] = 0

    def write_mha_metadata(
        self,
        gpu_view: ContextGPUView,
        query_lengths: Tensor,
        kv_lengths: Tensor,
        block_table: Tensor,
        padded_request_count: int,
    ) -> None:
        """Write the per-request MHA read metadata for one MTP forward.

        Args:
            gpu_view (ContextGPUView): Destination bookkeeping views.
            query_lengths (Tensor): [R] query rows contributed by each request.
            kv_lengths (Tensor): [R] KV entries each request attends over. May be the same
                tensor as `query_lengths` when the two coincide.
            block_table (Tensor): Block ids for the first `block_table.shape[0]` requests. Any
                remaining request slot -- the trailing pad request staged by
                `stage_prefill_lengths` -- is pointed at the dummy block.
            padded_request_count (int): Request slots the kernel launches over, including
                padding. Padded slots get zero lengths and a -1 block table, so they never index
                real KV.
        """
        n = query_lengths.numel()

        gpu_view.mha_query_lengths[:n] = query_lengths.to(gpu_view.mha_query_lengths.dtype)
        gpu_view.mha_cu_query_seq_lengths[0] = 0
        gpu_view.mha_cu_query_seq_lengths[1 : n + 1] = torch.cumsum(query_lengths, 0).to(
            gpu_view.mha_cu_query_seq_lengths.dtype
        )
        gpu_view.mha_kv_seq_lengths[:n] = kv_lengths.to(gpu_view.mha_kv_seq_lengths.dtype)
        gpu_view.mha_cu_kv_seq_lengths[0] = 0
        gpu_view.mha_cu_kv_seq_lengths[1 : n + 1] = torch.cumsum(kv_lengths, 0).to(
            gpu_view.mha_cu_kv_seq_lengths.dtype
        )

        num_real_rows = min(block_table.shape[0], n)
        gpu_view.mha_block_table[:num_real_rows] = block_table
        if n > num_real_rows:
            gpu_view.mha_block_table[num_real_rows:n] = self.dummy_block_idx

        if padded_request_count > n:
            gpu_view.mha_query_lengths[n:padded_request_count] = 0
            gpu_view.mha_cu_query_seq_lengths[n + 1 : padded_request_count + 1] = (
                gpu_view.mha_cu_query_seq_lengths[n]
            )
            gpu_view.mha_kv_seq_lengths[n:padded_request_count] = 0
            gpu_view.mha_cu_kv_seq_lengths[n + 1 : padded_request_count + 1] = (
                gpu_view.mha_cu_kv_seq_lengths[n]
            )
            gpu_view.mha_block_table[n:padded_request_count] = -1
