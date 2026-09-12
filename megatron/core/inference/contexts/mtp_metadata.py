# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
from torch import Tensor

from .gpu_view import ContextGPUView


@dataclass
class MTPMetadata:
    """Persistent metadata for the MTP (multi-token prediction) KV-cache forwards.

    A speculative step runs several MTP forwards back to back -- one per draft depth, plus a
    varlen "commit pass" -- and every one of them rewrites the same small set of per-request
    fields. Rather than build those tensors per forward, this class owns them as fixed-address
    buffers sized for the worst case and updates them in place. That keeps the draft loop
    allocation-free and CUDA-graph safe (a captured MTP graph replays against the same
    addresses each step).

    The buffers fall into two groups:
      * Draft-loop state (`offsets`, `block_table`): staged once per step by
        `begin_decode` and advanced per depth by `advance_decode_step`.
      * Per-forward staging (`query_lengths`, `kv_lengths`): rebuilt by `stage_*_lengths` for
        each forward, then written into the context's `gpu_view` by `write_mha_metadata`.
        They are staged here rather than written straight into `gpu_view` because the padded-row
        adjustments must be applied before the cumulative sums are taken.

    Construction is cheap and unconditional; `allocate` (called from the context's
    `initialize_all_tensors`) is what reserves GPU memory, and only when `enabled`.

    Args:
        enabled (bool): Whether MTP KV caching is active for this context. When False the
            object is inert: no buffers are allocated and `forward_active` stays False.
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

    # ---- Draft-loop state (valid between begin_decode() and end_decode()). ----
    # True while an MTP forward owns the attention metadata; the KV append/read paths key off
    # this to route themselves to the MTP layer slot instead of a main attention layer.
    forward_active: bool = False
    # Whether this draft loop replays captured CUDA graphs (True) or runs eager (False).
    graphed: bool = False
    active_request_count: int = 0
    padded_count: int = 0
    # `num_prefill_requests` saved across the commit pass, which forces the varlen path.
    saved_num_prefill_requests: int = 0
    # Block table as of just before `_rewind_kv_cache` released the draft blocks. None until
    # the first snapshot of the run.
    prerewind_block_table: Optional[Tensor] = field(default=None, repr=False)

    # ---- Chunked-prefill boundary carry (lives BETWEEN steps). ----
    # The MTP entry at position `off - 1` straddles two prefill chunks: it pairs the previous
    # chunk's last hidden with the next chunk's first token, so neither chunk can write it alone.
    # The producing chunk stashes its last hidden here; the consuming chunk splices it in. See
    # `_mtp_prefill_commit_segments`.
    #
    # The carry records the producing request's id AND the prompt position the hidden was computed
    # at, and `take_chunk_boundary` requires both to match, declining otherwise. The id alone is
    # not sufficient -- see that method for why.
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
    # Pinned CPU destination for the pre-rewind block-table snapshot.
    prerewind_buf: Optional[Tensor] = field(default=None, repr=False)

    # ---- Views into the buffers, refreshed once per draft loop by begin_decode(). ----
    active_offsets: Optional[Tensor] = field(default=None, repr=False)
    active_block_table: Optional[Tensor] = field(default=None, repr=False)

    def allocate(self, device: torch.device, block_table_template: Tensor) -> None:
        """Reserve the persistent buffers. No-op when MTP KV caching is disabled.

        Args:
            device (torch.device): Device for the GPU-resident buffers.
            block_table_template (Tensor): The context's CPU block table
                (`request_to_kv_block_ids`); the pre-rewind snapshot buffer mirrors its shape
                and dtype.
        """
        if not self.enabled:
            return
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
        self.prerewind_buf = torch.empty_like(block_table_template).pin_memory()
        self.chunk_boundary_hidden = torch.zeros(
            (1, 1, self.hidden_size), dtype=self.hidden_dtype, device=device
        )
        self.invalidate_chunk_boundary()

    def deallocate(self) -> None:
        """Release the persistent buffers, mirroring `allocate`.

        Used by the context's suspend path, which drops its tensors and rebuilds them from
        `initialize_all_tensors` on resume.
        """
        self.offsets = None
        self.block_table = None
        self.query_lengths = None
        self.kv_lengths = None
        self.row_ids = None
        self.prerewind_buf = None
        self.prerewind_block_table = None
        self.active_offsets = None
        self.active_block_table = None
        self.forward_active = False
        self.chunk_boundary_hidden = None
        self.invalidate_chunk_boundary()

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

    def take_chunk_boundary(self, req_id: int, seam_position: int) -> Optional[Tensor]:
        """Return the carried hidden if it is the one this seam needs, else None.

        BOTH keys must match, and a mismatch is declined rather than raised.

        A position mismatch should not happen: `_compute_prefix_match` gives up its ENTIRE prefix
        match for a continuation chunk of a request that still holds a carry, precisely so this
        chunk starts at `finished` and its seam lands where the carry describes. (Without that,
        a mid-request prefix match would move the chunk's start past the carried position and
        orphan that entry permanently.) This check is defence in depth for a path that is meant
        to be unreachable, so it declines rather than raising: skipping one seam costs a little
        draft acceptance and cannot affect verified output, whereas raising would kill a live
        step.

        The position key also subsumes the `off > 0` guard: a first chunk asks for
        `seam_position == -1`, and a valid carry always records a position `>= 0`.

        Args:
            req_id (int): Request that wants to write the seam.
            seam_position (int): MTP position the seam would be written at (`off - 1`).

        Returns:
            (Optional[Tensor]): The `[1, 1, hidden_size]` carried hidden, or None on any mismatch.
        """
        # The validity check must come first: the invalid sentinels are -1/-1, and an unfilled
        # request slot is also -1, so an invalid carry could otherwise match `req_id == -1` at a
        # first chunk's `seam_position == -1`.
        if not self.chunk_boundary_valid or self.chunk_boundary_req_id != req_id:
            return None

        if self.chunk_boundary_position != seam_position:
            # Logged rather than silent: this is legitimate (see above), but it is also what a
            # genuine bookkeeping drift would look like, and either way it costs draft acceptance.
            return None

        return self.chunk_boundary_hidden

    # ------------------------------------------------------------------
    # Draft-loop lifecycle.
    # ------------------------------------------------------------------
    def snapshot_prerewind_block_table(self, block_ids: Tensor) -> None:
        """Copy the current block table into the pre-rewind snapshot buffer."""
        if not self.enabled:
            return
        self.prerewind_buf.copy_(block_ids)
        self.prerewind_block_table = self.prerewind_buf

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
        self.forward_active = True

    def advance_decode_step(self) -> None:
        """Advance every active request's MTP write position by one, in place."""
        self.active_offsets += 1

    def end_forward(self) -> None:
        """Leave MTP-forward mode. No persistent length state to write back."""
        self.forward_active = False

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

        The pad rows are query rows as well as KV writes: the packed hidden is padded up to a
        TP multiple so it can be scattered for sequence parallelism, so after the MTP layer's
        internal gather the attention sees `total + pad_tokens` query rows while `append_counts`
        describes only `total` of them. Varlen attention requires `q.shape[0] == cu_seqlens_q[-1]`,
        so every pad row is given an owning request. Only the write maps decide where a forward's
        KV lands, and `write_token_maps` redirects the pad rows to the dummy block, so the
        extra query rows are inert wherever they are attributed.

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
            inherited_blocks (Optional[Tensor]): [R] leading blocks each row INHERITED from the
                prefix cache rather than computed. Tokens landing in those blocks are redirected
                to the dummy block: their KV is already correct from the producing request, and
                the block is ref-counted, so writing would corrupt every request sharing it.
                The main KV path redirects the same span (`overlap_start_token` in
                `add_request`). None disables the redirect, for callers with no inherited blocks.
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
