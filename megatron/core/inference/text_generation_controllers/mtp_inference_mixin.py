# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Multi-token-prediction (MTP) speculative decoding for dynamic inference.

`MTPInferenceMixin` holds the MTP half of `TextGenerationController`: the draft-KV commit pass,
the serial draft loop that produces speculative tokens, and the dummy forwards that keep idle
expert-parallel ranks in lockstep with them. It is a mixin rather than a standalone helper
because these paths read a large amount of controller state (sampling buffers, the wrapped
model, SP/PP topology); the split is for readability, and mixing in preserves the previous
behaviour exactly.

The only state the mixin writes is its own: the sampling/draft buffers allocated by
`_init_mtp_sampling_tensors`, and the chunked-prefill boundary hidden carried by
`_mtp_commit_pass`. Everything else it touches belongs to the inference context or the model.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from megatron.core.inference.communication_utils import broadcast_from_last_pipeline_stage
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.enums import InferenceCudaGraphScope
from megatron.core.transformer.moe.token_dispatcher_inference import NVLSAllGatherVDispatcher
from megatron.core.utils import nvtx_range_pop, nvtx_range_push, round_up_to_nearest_multiple


class MTPInferenceMixin:
    """MTP speculative-decoding paths for `TextGenerationController`."""

    def _init_mtp_sampling_tensors(self):
        """Pre-allocate MTP sampling tensors.

        Addresses must be stable across steps for CUDA graph capture.
        """
        self._mtp_resolved_padded_count = None
        if not self.num_speculative_tokens:
            self._sampled_mtp_tokens_cuda = None
            self._accepted_tokens_per_request = None
            self._last_accepted_seq_indices = None
            self._async_sched_mtp_token_row_indices = None
            self._async_sched_sampled_mtp_tokens_cpu_buffer = None
            self._async_sched_accepted_tokens_cpu_buffer = None
            self._async_sched_accepted_counts_cpu_buffer = None
            self._async_sched_mtp_verification_gpu_ready_event = None
            self._async_sched_accepted_counts_cpu_ready_event = None
            return

        context = self.inference_wrapped_model.inference_context
        max_requests = context.max_requests
        device = torch.cuda.current_device()
        self._sampled_mtp_tokens_cuda = torch.empty(
            [self.num_speculative_tokens, max_requests], dtype=torch.int64, device=device
        )
        self._async_sched_mtp_token_row_indices = torch.arange(context.max_tokens, device=device)
        self._accepted_tokens_per_request = (
            torch.ones(
                [max_requests, self.num_speculative_tokens], dtype=torch.int64, device=device
            )
            * -1
        )
        self._accepted_token_counts_per_request = torch.zeros(
            max_requests, dtype=torch.int64, device=device
        )
        self._last_accepted_seq_indices_buf = torch.empty(
            max_requests, dtype=torch.int64, device=device
        )
        self._last_accepted_seq_indices = None
        self._mtp_token_ids_buf = torch.empty([1, max_requests], dtype=torch.int64, device=device)
        self._mtp_position_ids_buf = torch.empty(
            [1, max_requests], dtype=torch.int64, device=device
        )
        self._async_sched_sampled_mtp_tokens_cpu_buffer = torch.empty(
            [self.num_speculative_tokens, max_requests],
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        )
        self._async_sched_accepted_tokens_cpu_buffer = torch.empty(
            [max_requests, self.num_speculative_tokens],
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        )
        self._async_sched_accepted_counts_cpu_buffer = torch.empty(
            max_requests, dtype=torch.int64, device="cpu", pin_memory=True
        )
        self._async_sched_mtp_verification_gpu_ready_event = torch.cuda.Event()
        self._async_sched_accepted_counts_cpu_ready_event = torch.cuda.Event()

    def _mtp_commit_pass(
        self,
        context,
        unwrapped_model,
        gathered_hidden,
        num_decode_requests: int,
        active_request_count: int,
        base_position,
    ) -> bool:
        """Refresh every committed position's draft KV from the MAIN hidden (a per-step first pass).

        One varlen roll-by-one forward over all active requests' committed tokens, so a committed
        position p always holds draft KV = f(h_p^main + emb(t_{p+1})) and never a stale
        chained-draft-hidden value (the cause of depth-increasing acceptance decay). K/V are pure
        projections of the input (no RoPE), so only the write positions matter; the output hidden
        is discarded. Covers, per request:
          - decode: the a_r accepted-draft positions this step (base + drafts 0..a-2), from this
            step's main hiddens for the accepted forwarded tokens. The LAST accepted
            position is written by decode depth 0, so this covers the earlier a_r
            positions (start = base-1-a_r).
          - prefill: the chunk's positions off..off+q-2 (start = off = request_kv_length_offset);
            for a continuation chunk (off > 0) the straddling position off-1 is also seeded from the
            carried-over previous-chunk hidden. off == 0 and q == full prompt length in the common
            (single-chunk) case.
        Returns True if it issued a forward (caller runs a dummy slot otherwise for EP balance).
        """
        device = gathered_hidden.device
        stride = self.num_speculative_tokens + 1
        decode_len = num_decode_requests * stride
        active_slice = slice(context.paused_request_count, context.total_request_count)
        chunked_id = context.chunked_prefill_request_id

        hidden_parts, token_parts, append_parts, start_parts = [], [], [], []

        # --- Decode requests: refresh accepted-draft positions from main hiddens. ---
        if num_decode_requests > 0:
            accepted = (
                self._accepted_token_counts_per_request[:num_decode_requests]
                .to(device, non_blocking=True)
                .to(torch.long)
            )
            append_parts.append(accepted)
            # start of each request's refreshed run: (last committed MTP pos) - a_r.
            start_parts.append(base_position[:num_decode_requests] - 1 - accepted)
            total_a = int(accepted.sum().item())
            if total_a > 0:
                decode_hidden = gathered_hidden[:decode_len]  # [num_decode*stride, 1, H]
                decode_tokens = context.gpu_view.token_to_input_ids[:decode_len].to(device)
                rows_d = torch.repeat_interleave(
                    torch.arange(num_decode_requests, device=device), accepted
                )
                within_d = (
                    torch.arange(total_a, device=device)
                    - (torch.cumsum(accepted, 0) - accepted)[rows_d]
                )
                hidden_idx = rows_d * stride + within_d  # base..draft_{a-2} main hiddens
                hidden_parts.append(decode_hidden[hidden_idx])
                token_parts.append(decode_tokens[hidden_idx + 1])  # draft_0..draft_{a-1}

        # --- Prefill requests: seed their prompt positions from main hiddens (roll-by-one). ---
        # request_query_lengths is this step's CHUNK length `q` and request_kv_length_offsets is how
        # many prompt tokens prior chunks already prefilled (`off`), so each request seeds positions
        # off..off+q-2 (its last chunk position off+q-1 is seeded next chunk, or by the decode draft
        # loop on the final/only chunk). For a continuation chunk (off > 0) the position straddling
        # the previous chunk, off-1, is also seeded here from the carried-over previous-chunk hidden
        # h_{off-1} paired with this chunk's first token t_off. Built per request (num_prefill
        # is small and this forward is eager) so the continuation stays ONE segment -> the
        # segment count stays active_request_count and the fixed-size MHA-metadata buffers
        # never overflow.
        # request_* are CPU pinned tensors, so `.tolist()` is a cheap host read (no GPU sync).
        if active_request_count > num_decode_requests:
            prefill_slice = slice(num_decode_requests, active_request_count)
            q_list = context.request_query_lengths[active_slice][prefill_slice].tolist()
            off_list = context.request_kv_length_offsets[active_slice][prefill_slice].tolist()
            id_list = context.request_ids[active_slice][prefill_slice].tolist()
            total_prompt = sum(q_list)
            prefill_hidden = gathered_hidden[decode_len : decode_len + total_prompt]
            prefill_tokens = context.gpu_view.token_to_input_ids[
                decode_len : decode_len + total_prompt
            ].to(device)

            p_counts, p_starts, p_hidden_segs, p_token_segs = [], [], [], []
            cum = 0
            for q, off, rid in zip(q_list, off_list, id_list):
                h_i = prefill_hidden[cum : cum + q]  # [q, 1, H] this chunk's hiddens
                t_i = prefill_tokens[cum : cum + q]  # [q]     this chunk's tokens
                cum += q
                # h_{off-1} is needed to seed the straddling entry at off-1, and is available only
                # when this request computed position off-1 itself, i.e. off came from its own
                # previous chunk. A prefix-cache hit also produces off > 0, but there the skipped
                # prefix was computed by a DIFFERENT request whose activations are gone -- see the
                # `else` branch.
                own_prior_chunk = (
                    self._mtp_chunk_boundary_hidden is not None
                    and self._mtp_chunk_boundary_req_id == rid
                )
                if off > 0 and own_prior_chunk:
                    # Continuation chunk: boundary position off-1 = f(carried h_{off-1} + t_off),
                    # then this chunk's roll-by-one off..off+q-2. count = q, start = off-1.
                    p_hidden_segs.append(self._mtp_chunk_boundary_hidden.to(device))
                    p_hidden_segs.append(h_i[:-1])
                    p_token_segs.append(t_i[:1])
                    p_token_segs.append(t_i[1:])
                    p_counts.append(q)
                    p_starts.append(off - 1)
                else:
                    # off == 0 (nothing precedes), or a prefix-cache hit. On a hit this request
                    # inherits the matched blocks' draft KV, which is already correct for every
                    # position p <= off-2: that entry is f(h_p + emb(t_{p+1})) and t_{p+1} lies
                    # inside the shared prefix, so it is the same token for every request sharing
                    # the block. Only position off-1 differs, because its entry consumes t_off --
                    # the first token where children diverge. That entry is NOT rewritten here: it
                    # lives in a ref-counted block shared with the producer and every sibling, so
                    # writing this request's value would corrupt theirs. One stale key/value at the
                    # divergence point costs a little draft acceptance and cannot affect verified
                    # output. count = q-1, start = off.
                    p_hidden_segs.append(h_i[:-1])
                    p_token_segs.append(t_i[1:])
                    p_counts.append(q - 1)
                    p_starts.append(off)
                # Carry this step's in-flight chunk's last hidden for the next chunk's boundary.
                if chunked_id != -1 and rid == chunked_id:
                    self._mtp_chunk_boundary_hidden = h_i[-1].detach().clone().view(1, 1, -1)
                    self._mtp_chunk_boundary_req_id = chunked_id

            append_parts.append(torch.tensor(p_counts, dtype=torch.long, device=device))
            start_parts.append(torch.tensor(p_starts, dtype=torch.long, device=device))
            hidden_parts.append(torch.cat(p_hidden_segs))
            token_parts.append(torch.cat(p_token_segs))

        # No in-flight chunked request (none, or it finished its final chunk this step): drop the
        # carried boundary hidden so a later request can never match a stale id.
        if chunked_id == -1:
            self._mtp_chunk_boundary_hidden = None
            self._mtp_chunk_boundary_req_id = -1

        append_counts = torch.cat(append_parts)
        request_start_positions = torch.cat(start_parts)
        total = int(append_counts.sum().item())
        if total == 0:
            # Nothing committed to (re)write this step (e.g. all decode requests accepted 0 drafts
            # and no prefill). Caller runs a dummy slot so the EP forward count stays matched.
            return False

        packed_hidden = torch.cat(hidden_parts)  # [total, 1, H], decode-first request order
        packed_tokens = torch.cat(token_parts)  # [total]
        block_table = (
            context.request_to_kv_block_ids[active_slice][:active_request_count]
            .to(device, non_blocking=True)
            .to(context.gpu_view.mha_block_table.dtype)
        )

        padded_total = total
        if self._sp_enabled:
            padded_total = round_up_to_nearest_multiple(total, self._tp_size)
            packed_hidden = F.pad(packed_hidden, (0, 0, 0, 0, 0, padded_total - total))
            packed_hidden = scatter_to_sequence_parallel_region(
                packed_hidden, group=self.inference_wrapped_model.tp_group
            )

        context._mtp_setup_prefill_step(
            append_counts=append_counts,
            block_table_prefill=block_table,
            padded_token_count=padded_total,
            padded_request_count=active_request_count,
            request_start_positions=request_start_positions,
        )
        if context._nvls_dispatcher:
            NVLSAllGatherVDispatcher.modify_real_token_count_for_mtp(total)

        token_ids = packed_tokens.view(1, -1).to(torch.long)
        # Positions are unused for attention (no RoPE); the write positions come from the metadata.
        position_ids = torch.zeros_like(token_ids)
        if padded_total > total:
            token_ids = F.pad(token_ids, (0, padded_total - total))
            position_ids = F.pad(position_ids, (0, padded_total - total))

        # Run the MTP attention to populate K/V only (output hidden discarded).
        unwrapped_model.mtp.layers[0].forward_single_position(
            hidden_states=packed_hidden,
            next_token_ids=token_ids,
            position_ids=position_ids,
            embedding=unwrapped_model.embedding,
            inference_context=context,
        )
        context._mtp_finalize_prefill_step()
        return True

    def _mtp_dummy_prefill_forward(self, context, unwrapped_model) -> None:
        """Issue one MTP-layer forward with dummy tensors and NO KV append.

        Every rank must run exactly one MTP "prefill-slot" forward per step so the MoE/EP
        all-to-alls stay balanced when some ranks seed a real prompt and others (idle ranks,
        or active ranks with no prefill this step) do not. Uses ``inference_context=None`` so
        the attention runs cache-free (no child append); only the MoE all-to-all matters here.
        Mirrors the real seed's single ``forward_single_position`` call.
        """
        device = torch.cuda.current_device()
        dtype = self.model_config.params_dtype
        hidden_size = self.model_config.hidden_size
        n = self._tp_size if self._sp_enabled else 1
        dummy_hidden = torch.zeros((n, 1, hidden_size), device=device, dtype=dtype)
        if self._sp_enabled:
            dummy_hidden = scatter_to_sequence_parallel_region(
                dummy_hidden, group=self.inference_wrapped_model.tp_group
            )
        dummy_tokens = torch.zeros((1, n), device=device, dtype=torch.long)
        dummy_positions = torch.zeros((1, n), device=device, dtype=torch.long)
        unwrapped_model.mtp.layers[0].forward_single_position(
            hidden_states=dummy_hidden,
            next_token_ids=dummy_tokens,
            position_ids=dummy_positions,
            embedding=unwrapped_model.embedding,
            inference_context=None,
        )

    def _compute_serial_mtp_and_sample(self, base_position: Optional[Tensor] = None) -> None:
        """Compute MTP logits serially after verification and sample speculative tokens.

        This ensures that MTP predictions are always conditioned on verified tokens.
        Each MTP depth receives the correctly sampled token from the previous depth
        (or the base token for depth 0) rather than stale speculative tokens from
        the previous step.

        When sequence parallelism is active, hidden states are kept in SP format
        (scattered along the first dimension) between MTP depths to avoid a
        redundant gather + scatter round-trip per depth.

        Args:
            base_position (Optional[Tensor]): GPU position of the first new MTP draft
                for each request. Legacy scheduling derives it from rewound CPU state.
        """
        nvtx_range_push("mtp-spec-decoding/serial-mtp-init")
        context = self.inference_wrapped_model.inference_context
        active_request_count = context.total_request_count - context.paused_request_count
        active_slice = slice(context.paused_request_count, context.total_request_count)

        unwrapped_model = self._unwrapped_model

        # On non-last pipeline stages, the model won't have decoder hidden states.
        has_mtp = self._is_last_pp_stage and context.mtp_decoder_hidden_states is not None

        if has_mtp:
            # Get decoder hidden states at last accepted positions.
            hidden_states = context.mtp_decoder_hidden_states

            # Block-scope CUDA graphs write into a persistent max_tokens-sized
            # buffer. Only the prefix for this step is valid. Slice each rank's
            # local SP shard before gathering; gathering the oversized buffer
            # would place rank 0's stale tail between the valid rank shards.
            if context.inference_cuda_graph_scope == InferenceCudaGraphScope.block:
                local_token_count = context.padded_active_token_count
                if self._sp_enabled:
                    assert local_token_count % self._tp_size == 0
                    local_token_count //= self._tp_size
                hidden_states = hidden_states[:local_token_count]

            # When SP is active the decoder output is in scattered format
            # [S/TP, B, H], but _last_accepted_seq_indices are indices into
            # the full (gathered) sequence.
            if self._sp_enabled:
                hidden_states = gather_from_sequence_parallel_region(
                    hidden_states, group=self.inference_wrapped_model.tp_group
                )
            last_accepted_hidden = hidden_states[self._last_accepted_seq_indices, :, :]
            # Shape: [active_request_count, 1, hidden_size]
        else:
            last_accepted_hidden = None

        # MTP KV cache (v1): seed the draft KV from the prompt for any request prefilling this
        # step (roll-by-one), before the decode draft loop reads/extends it. Uses the gathered
        # decoder hidden states above; runs only on the last PP stage where they exist.
        if base_position is None:
            # Legacy scheduling derives positions from post-rewind CPU state.
            # After rewind, request_kv_length_offsets has been adjusted. Read from
            # CPU context (post-rewind values), NOT gpu_view (stale pre-rewind snapshot).
            # The next position to predict is: adjusted_offset + processed_tokens.
            cuda_device = torch.cuda.current_device()
            adjusted_offsets = context.request_kv_length_offsets[active_slice].to(
                cuda_device, non_blocking=True
            )
            processed_tokens = context.request_query_lengths[active_slice].to(
                cuda_device, non_blocking=True
            )
            # Cast to int64 to match CUDA graph capture dtype expectations.
            base_position = (adjusted_offsets + processed_tokens).to(torch.int64)

        # MTP KV cache (v1): before the draft loop, refresh every committed position's draft KV
        # from the MAIN model hidden states (a per-step "first pass"), so committed KV never
        # carries a stale chained-draft-hidden value (which would decay acceptance with depth).
        # Covers prefill prompts (seed) and decode accepted drafts (refresh) in one forward.
        if getattr(context, "enable_mtp_kv_cache", False) and has_mtp:
            issued_slot = self._mtp_commit_pass(
                context,
                unwrapped_model,
                hidden_states,
                num_decode_requests=active_request_count - context.num_prefill_requests,
                active_request_count=active_request_count,
                base_position=base_position,
            )
            if not issued_slot:
                # EP consistency: every rank runs exactly one commit-pass forward per step (real
                # above, or this dummy) so the MoE all-to-alls stay count-matched.
                self._mtp_dummy_prefill_forward(context, unwrapped_model)

        # Start with the freshly sampled base token.
        next_token_ids = self._sampled_tokens_cuda[:active_request_count].clone()
        current_hidden = last_accepted_hidden if has_mtp else None

        # Compute padding needed to make batch compatible with SP and CUDA graphs.
        if self._mtp_resolved_padded_count is not None:
            # CUDA-graph path: use the EP-synced padded count.
            padded_count = self._mtp_resolved_padded_count
            assert not self._sp_enabled or padded_count % self._tp_size == 0
        elif has_mtp:
            # Eager path: pad only for SP alignment.
            padded_count = active_request_count
            if self._sp_enabled:
                padded_count = round_up_to_nearest_multiple(padded_count, self._tp_size)
        else:
            padded_count = active_request_count
        pad_count = padded_count - active_request_count

        # Pad hidden states and scatter for sequence parallelism.
        if has_mtp:
            current_hidden = F.pad(current_hidden, (0, 0, 0, 0, 0, pad_count))
            if self._sp_enabled:
                current_hidden = scatter_to_sequence_parallel_region(
                    current_hidden, group=self.inference_wrapped_model.tp_group
                )

        token_ids_buf = self._mtp_token_ids_buf[:, :padded_count]
        position_ids_buf = self._mtp_position_ids_buf[:, :padded_count]

        # Zero-fill padding slots so the embedding layer never sees out-of-range IDs.
        token_ids_buf[0, active_request_count:] = 0
        position_ids_buf[0, active_request_count:] = 0

        nvtx_range_pop("mtp-spec-decoding/serial-mtp-init")

        # MTP MoE forwards are request-count shaped: the routing map holds
        # active_request_count real rows followed by padding up to padded_count.
        # The NVLS routing mask defaults to the main step's token count, so point
        # it at the MTP row count instead, else padding rows route to experts.
        if context._nvls_dispatcher:
            NVLSAllGatherVDispatcher.modify_real_token_count_for_mtp(active_request_count)

        # MTP KV cache (v1): give the draft attention its own KV in the shared buffer's reserved
        # slot. Each depth is a decode-style forward (one token per active request). The depth-0
        # write position is base_position - 1 (roll-by-one): the MTP entry for main position
        # base_position-1 is computed from H_{base_position-1} + emb(base token). Deriving it from
        # base_position each step (rather than tracking a separate MTP length) means it can never
        # desync through compaction/pause/rewind. `_mtp_advance_decode_step` bumps it per depth.
        mtp_kv_cache_on = getattr(context, "enable_mtp_kv_cache", False) and has_mtp
        # Whether the MTP draft forwards replay captured CUDA graphs this step. Mirror the MAIN
        # decode step's graph decision via `_mtp_resolved_padded_count` (set from the un-clobbered,
        # EP-synced graph flag right after the main forward; None iff the main step was eager). This
        # is the same signal the EP dummy path uses, so real and dummy ranks stay in lockstep.
        mtp_graphed = mtp_kv_cache_on and self._mtp_resolved_padded_count is not None
        # KV-aware MTP graphs are captured under a distinct key ("mtp_kv") from the cache-free
        # graphs ("mtp") that the normal spec-decode path and the EP dummy path replay. The dummy
        # rank cannot safely replay the KV-aware graph (its append would write the idle rank's KV
        # cache with no valid block table), so it uses the cache-free graph; both still issue
        # identical fixed-size (expert-padded) MoE all-to-alls, keeping EP in lockstep.
        mtp_graph_key_prefix = "mtp_kv" if mtp_kv_cache_on else "mtp"
        # A still-prefilling chunked request (chunked_prefill_request_id != -1) is always the last
        # active request and has base_position mid-prompt, so it must not draft — it decodes only
        # once its prompt completes. Excluding it (reducing the draft count by 1) makes it a padding
        # row in _mtp_setup_decode_step (no draft KV write); its chunk KV was already seeded by the
        # commit pass. padded_count (the graph size) and the MTP-forward count are unchanged, so the
        # captured graph, EP parity, and dummy path are unaffected.
        num_mtp_draft_requests = active_request_count - (
            1 if context.chunked_prefill_request_id != -1 else 0
        )
        if mtp_kv_cache_on:
            context._mtp_begin_decode(
                num_mtp_draft_requests, padded_count, base_position - 1, graphed=mtp_graphed
            )

        for depth in range(self.num_mtp_depths):
            nvtx_range_push(f"mtp-spec-decoding/depth-{depth}")

            token_ids_buf[0, :active_request_count] = next_token_ids
            position_ids_buf[0, :active_request_count] = base_position + depth

            mtp_logits_2d = None
            if has_mtp:
                nvtx_range_push(f"mtp-spec-decoding/depth-{depth}/forward")
                mtp_depth = None if unwrapped_model.mtp.mtp_use_repeated_layer else depth
                if mtp_kv_cache_on:
                    context._mtp_setup_decode_step()
                current_hidden, mtp_logits = unwrapped_model.compute_mtp_single_step(
                    hidden_states=current_hidden,
                    next_token_ids=token_ids_buf,
                    position_ids=position_ids_buf,
                    depth=mtp_depth,
                    eager=not context.using_cuda_graph_this_step(),
                    cache_key=(
                        (mtp_graph_key_prefix, padded_count, mtp_depth)
                        if context.using_cuda_graph_this_step()
                        else None
                    ),
                    mtp_inference_context=context if mtp_kv_cache_on else None,
                )
                if mtp_kv_cache_on:
                    context._mtp_advance_decode_step()
                nvtx_range_pop(f"mtp-spec-decoding/depth-{depth}/forward")

                # Strip padding from logits only. Hidden states stay padded+SP
                # between depths to avoid redundant gather/scatter round-trips.
                mtp_logits = mtp_logits[:active_request_count]

                # mtp_logits: [active_request_count, 1, vocab_size]
                mtp_logits_2d = mtp_logits.squeeze(1)  # [active_request_count, vocab_size]

            # Broadcast MTP logits across pipeline stages.
            if self.model_is_pipeline_parallel:
                nvtx_range_push(f"mtp-spec-decoding/depth-{depth}/pp-broadcast")
                mtp_logits_2d = broadcast_from_last_pipeline_stage(
                    [active_request_count, self.vocab_size],
                    dtype=self.model_config.params_dtype,
                    tensor=mtp_logits_2d,
                    pp_group=self.pp_group,
                )
                nvtx_range_pop(f"mtp-spec-decoding/depth-{depth}/pp-broadcast")

            # Sample speculative token using the same sampling parameters.
            nvtx_range_push(f"mtp-spec-decoding/depth-{depth}/sample")
            spec_tokens = self._sample_from_logits_2d(mtp_logits_2d)
            self._sampled_mtp_tokens_cuda[depth, :active_request_count] = spec_tokens
            nvtx_range_pop(f"mtp-spec-decoding/depth-{depth}/sample")

            # Use sampled token as input for the next depth.
            next_token_ids = spec_tokens
            nvtx_range_pop(f"mtp-spec-decoding/depth-{depth}")

        # MTP KV cache: append one extra entry for the final draft (spec[D-1]). The loop ran
        # D depths (base + spec[0..D-2]); the main model forwards all D+1 positions next step
        # (base + spec[0..D-1]), so the MTP KV must hold D+1 too, else it is one short at full
        # acceptance. This forward only populates K/V (its logits feed no further depth). No
        # explicit MTP rewind is needed: next step derives its start from base_position (which
        # advances by 1 + accepted) and overwrites the rejected drafts' positions.
        if mtp_kv_cache_on and has_mtp:
            token_ids_buf[0, :active_request_count] = next_token_ids
            position_ids_buf[0, :active_request_count] = base_position + self.num_mtp_depths
            context._mtp_setup_decode_step()
            extra_depth = (
                None if unwrapped_model.mtp.mtp_use_repeated_layer else self.num_mtp_depths - 1
            )
            # The extra-append is a structurally identical one-token append+attend, so it reuses the
            # last depth's captured KV-aware graph key (repeated-layer -> ("mtp_kv", n, None);
            # otherwise ("mtp_kv", n, D-1)); both are captured during warmup. Leaving cache_key=None
            # while graphed would trigger an illegal runtime capture.
            unwrapped_model.compute_mtp_single_step(
                hidden_states=current_hidden,
                next_token_ids=token_ids_buf,
                position_ids=position_ids_buf,
                depth=extra_depth,
                eager=not mtp_graphed,
                cache_key=(("mtp_kv", padded_count, extra_depth) if mtp_graphed else None),
                mtp_inference_context=context,
            )
            context._mtp_advance_decode_step()

        if mtp_kv_cache_on:
            context._mtp_end_decode()

        # In eager mode forward() assigns the hidden states tensor directly to
        # the context attribute; release it so the tensor can be garbage
        # collected. In block-scope CUDA graph mode the attribute is a
        # pre-allocated fixed buffer that must persist across replays.
        if has_mtp and context.inference_cuda_graph_scope != InferenceCudaGraphScope.block:
            context.mtp_decoder_hidden_states = None

    @torch.inference_mode()
    def _run_dummy_serial_mtp_forward(self) -> None:
        """Run dummy MTP forward passes to participate in EP collectives.

        When speculative decoding is active and MTP layers contain MoE sublayers
        (inherited from the decoder layer spec), each serial MTP step triggers
        EP all-to-all collectives. The dummy EP rank must issue matching
        collective calls so the real ranks do not hang.

        This mirrors the structure of ``_compute_serial_mtp_and_sample``:
        - On the last PP stage (where MTP resides): run ``compute_mtp_single_step``
          with dummy tensors so the MoE all-to-all is executed.
        - When PP > 1: participate in the ``broadcast_from_last_pipeline_stage``
          that the real ranks also perform.
        """
        if self.num_speculative_tokens == 0 or self.num_mtp_depths == 0:
            return
        if self.model_config.expert_model_parallel_size <= 1:
            return

        context = self.inference_wrapped_model.inference_context
        unwrapped_model = self._unwrapped_model
        has_mtp = self._is_last_pp_stage and hasattr(unwrapped_model, "mtp")
        if not has_mtp and not self.model_is_pipeline_parallel:
            # No MTP on this rank and no PP broadcast to participate in.
            return

        device = torch.cuda.current_device()
        dtype = self.model_config.params_dtype
        hidden_size = self.model_config.hidden_size

        # Use precomputed MTP CUDA graph batch size when available;
        # otherwise use minimal SP-compatible size.
        if self._mtp_resolved_padded_count is not None:
            padded_count = self._mtp_resolved_padded_count
            assert not self._sp_enabled or padded_count % self._tp_size == 0
        elif has_mtp:
            # Eager path: use TP-aligned minimum size for dummy tensors.
            padded_count = self._tp_size if self._sp_enabled else 1

        dummy_hidden = None
        if has_mtp:
            # Minimal dummy tensors to drive the MTP layer forward
            # so that the MoE all-to-all collectives are issued.
            dummy_hidden = torch.zeros((padded_count, 1, hidden_size), device=device, dtype=dtype)
            if self._sp_enabled:
                dummy_hidden = scatter_to_sequence_parallel_region(
                    dummy_hidden, group=self.inference_wrapped_model.tp_group
                )
            dummy_token_ids = torch.zeros((1, padded_count), device=device, dtype=torch.long)
            dummy_position_ids = torch.zeros((1, padded_count), device=device, dtype=torch.long)

        context = self.inference_wrapped_model.inference_context

        # When the MTP KV cache is active, real ranks run one prefill-slot forward BEFORE the depth
        # loop and one extra-append forward AFTER it. The dummy EP rank must mirror both the COUNT
        # (D+2 forwards) and the graph/eager MODE, else the shared MoE all-to-all count/shape
        # mismatches and the collective hangs. Mirror the real path's mode via the EP-synced
        # `_mtp_resolved_padded_count` (None iff the main step was eager) -- NOT the local live
        # `using_cuda_graph_this_step()`, which the commit pass clobbers on the real ranks.
        #
        # The dummy always replays the CACHE-FREE ("mtp", ...) MTP graph, never the KV-aware
        # ("mtp_kv", ...) one: replaying the KV-aware graph here would run its append against the
        # idle rank's KV cache with no valid block table (OOB). The cache-free graph has the same
        # fixed-size MoE all-to-all footprint, so EP stays matched -- this is exactly the (working)
        # dummy path used by the normal non-KV spec-decode flow.
        mtp_cache_active = getattr(context, "enable_mtp_kv_cache", False)
        main_graphed = getattr(self, "_mtp_resolved_padded_count", None) is not None
        mtp_forward_eager = not main_graphed
        if mtp_cache_active and has_mtp:
            self._mtp_dummy_prefill_forward(context, unwrapped_model)

        for depth in range(self.num_mtp_depths):
            nvtx_range_push(f"mtp-spec-decoding/dummy-depth-{depth}")
            mtp_logits_2d = None
            if has_mtp:
                mtp_depth = None if unwrapped_model.mtp.mtp_use_repeated_layer else depth
                dummy_hidden, mtp_logits = unwrapped_model.compute_mtp_single_step(
                    hidden_states=dummy_hidden,
                    next_token_ids=dummy_token_ids,
                    position_ids=dummy_position_ids,
                    depth=mtp_depth,
                    eager=mtp_forward_eager,
                    cache_key=(("mtp", padded_count, mtp_depth) if not mtp_forward_eager else None),
                )
                mtp_logits_2d = mtp_logits.squeeze(1)  # [padded_count, vocab_size]

            # Match the PP broadcast that real ranks do in _compute_serial_mtp_and_sample.
            if self.model_is_pipeline_parallel:
                broadcast_from_last_pipeline_stage(
                    [padded_count, self.vocab_size],
                    dtype=dtype,
                    tensor=mtp_logits_2d,
                    pp_group=self.pp_group,
                )
            nvtx_range_pop(f"mtp-spec-decoding/dummy-depth-{depth}")

        # Extra-append forward to match the real path's D+1th MTP forward (no PP broadcast,
        # mirroring the real extra append which does not broadcast its logits).
        if mtp_cache_active and has_mtp:
            extra_depth = (
                None if unwrapped_model.mtp.mtp_use_repeated_layer else self.num_mtp_depths - 1
            )
            unwrapped_model.compute_mtp_single_step(
                hidden_states=dummy_hidden,
                next_token_ids=dummy_token_ids,
                position_ids=dummy_position_ids,
                depth=extra_depth,
                eager=mtp_forward_eager,
                cache_key=(("mtp", padded_count, extra_depth) if not mtp_forward_eager else None),
            )
