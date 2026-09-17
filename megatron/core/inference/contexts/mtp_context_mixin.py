# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Context-side half of the MTP draft-KV cache.

`MTPContextMixin` holds the per-forward bookkeeping `DynamicInferenceContext` performs for the
MTP draft attention: entering draft-forward mode, publishing the write maps and read metadata
for one depth or for the varlen commit pass, and leaving again. It also owns the gate that
decides whether this model/config can populate a draft KV plane at all.
"""

import contextlib
from typing import List, Optional

import torch
from torch import Tensor

from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from megatron.core.transformer.enums import InferenceCudaGraphScope

from .attention_context.mha_metadata import MHAMetadata
from .mtp_metadata import MTPForwardMode


class MTPContextMixin:
    """MTP draft-KV bookkeeping for `DynamicInferenceContext`."""

    @staticmethod
    def should_enable_mtp_kv_cache(
        model_config, mtp_layer_type_list: Optional[List[str]], num_speculative_tokens: int
    ) -> bool:
        """Whether this model/config can populate an MTP draft-KV plane.

        Args:
            model_config: Transformer config, read for the MTP head's shape.
            mtp_layer_type_list (Optional[List[str]]): Layer types of one MTP draft-head depth
                for a hybrid model; None for a non-hybrid one.
            num_speculative_tokens (int): Draft depth; 0 disables speculative decoding.

        Returns:
            (bool): True when the draft KV plane should be reserved.
        """
        # The draft plane reserves exactly one attention slot, so the head must be one layer.
        # `mtp_use_repeated_layer` is what guarantees that: it builds a single layer and applies
        # it at every depth, where the non-repeated path builds `mtp_num_layers` of them.
        if not (
            num_speculative_tokens > 0
            and getattr(model_config, "mtp_num_layers", None)
            and getattr(model_config, "mtp_use_repeated_layer", False)
        ):
            return False

        if mtp_layer_type_list is None:
            # Non-hybrid model -- `HybridModel` builds a head only when the pattern has an MTP
            # section, so a hybrid head always exposes one. That layer is cloned from the last
            # decoder layer spec, so it is attention by construction.
            return True

        # The reserved slot holds one layer's KV, so a head with a second attention layer would
        # have both depths read and write the same plane.
        attention_symbols = (Symbols.ATTENTION, Symbols.DS_ATTENTION, Symbols.MLA)
        num_attention = sum(t in attention_symbols for t in mtp_layer_type_list)
        has_recurrent = any(t in (Symbols.MAMBA, Symbols.GDN) for t in mtp_layer_type_list)
        return num_attention == 1 and not has_recurrent

    # ------------------------------------------------------------------
    # MTP draft-KV bookkeeping.
    #
    # The draft attention reuses this context's KV `memory_buffer` (slot `mtp_kv_layer_slot`)
    # and main's block table. Each draft depth is a decode-style forward: one token per active
    # request, written at its own draft position and attending over its own history
    # (write-then-attend). RoPE is assumed absent.
    #
    # There is NO persistent per-request draft length. Depth 0's write position is DERIVED each
    # step as `base_position - 1` (roll-by-one). Because the main KV offsets it comes from are
    # already maintained through compaction, pause/resume and speculative rewind, the derived
    # position cannot desync -- and no separate draft rewind is needed, since `base_position`
    # advances by exactly 1 + accepted and rejected drafts are overwritten next step.
    #
    # The metadata is driven directly on the GPU, bypassing the coalesced CPU->GPU bookkeeping
    # transfer, so draft forwards never disturb the main step's Mamba/H2D state.
    # ------------------------------------------------------------------
    @contextlib.contextmanager
    def _mtp_forward_phase(self):
        """Scope one step's MTP forwards and undo their effects on the context on the way out.

        Every MTP forward republishes the active attention metadata, token counts and CUDA-graph
        flag, and leaves `forward_mode` set so KV append/read route to the draft plane. Log-prob
        computation runs after this scope and reads those fields expecting the MAIN forward's
        values, and a `forward_mode` left set would keep `is_decode_only` False and KV routing on
        the draft plane for the rest of the run. Both are undone on exit, including when a draft
        forward raises.
        """
        saved = (
            self.active_attn_metadata,
            self.active_token_count,
            self.padded_active_token_count,
            self._using_cuda_graph_this_step,
        )
        try:
            yield
        finally:
            self.mtp_metadata.end_forward()
            # In eager mode `forward()` assigns the hidden states straight to the context
            # attribute; release it so the tensor can be collected. Under block-scope CUDA graphs
            # the attribute is a pre-allocated buffer that must persist across replays.
            if self.inference_cuda_graph_scope != InferenceCudaGraphScope.block:
                self.mtp_decoder_hidden_states = None
            (
                self.active_attn_metadata,
                self.active_token_count,
                self.padded_active_token_count,
                self._using_cuda_graph_this_step,
            ) = saved

    def _mtp_activate_attn_metadata(
        self,
        graphed: bool,
        padded_request_count: int,
        max_seqlen_q: int,
        max_seqlen_k: int,
        token_count: int,
        padded_token_count: int,
    ) -> MHAMetadata:
        """Publish the attention metadata for one MTP forward and return the active MHA object.

        The buffer writes above target the fixed-address `gpu_view` buffers that the append/attend
        kernels read; those buffers are shared by the graphed and non-graphed MHA metadata objects,
        so a captured MTP graph replays correctly against whatever positions were just written.
        Only the metadata object selected here, the sequence-length bounds, and the token counts
        vary per step.
        """
        attn_metadata = self.graph_attn_metadata if graphed else self.non_graph_attn_metadata
        mha = attn_metadata["mha_metadata"]
        mha.set_state_data(
            padded_active_request_count=padded_request_count,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
        )
        self.active_attn_metadata = attn_metadata
        self.active_token_count = token_count
        self.padded_active_token_count = padded_token_count
        self._using_cuda_graph_this_step = graphed
        return mha

    def _mtp_begin_decode(
        self,
        active_request_count: int,
        padded_count: int,
        start_positions: Tensor,
        graphed: bool = False,
    ) -> None:
        """Enter MTP-forward mode.

        `start_positions[r] == base_position[r] - 1` is the MTP write position for depth 0 of
        request r (derived from the main KV offsets by the caller), advanced by one per depth.

        `graphed` mirrors the main decode step's CUDA-graph decision (the caller passes it from the
        EP-synced `_mtp_resolved_padded_count`, NOT the live `_using_cuda_graph_this_step` which the
        commit pass has already clobbered). When True, `_mtp_setup_decode_step` routes to the graph
        attention metadata so the captured KV-aware MTP graph is replayed.
        """
        assert self.enable_mtp_kv_cache
        active_slice = slice(self.paused_request_count, self.total_request_count)
        # Rewind retains MTP lookahead blocks, so the live table owns every draft write.
        # Reading the live row also follows compaction and sync/async scheduler transitions.
        self.mtp_metadata.begin_decode(
            active_request_count=active_request_count,
            padded_count=padded_count,
            start_positions=start_positions,
            block_table_src=self.request_to_kv_block_ids[active_slice],
            graphed=graphed,
        )

    def _mtp_setup_decode_step(self) -> None:
        """Populate token write maps + MHA read metadata for one MTP draft depth."""
        mtp = self.mtp_metadata
        n = mtp.active_request_count
        padded = mtp.padded_count

        positions = mtp.active_offsets  # [n] int, MTP write position P_r for this depth

        # Token write maps (one token per active request), padded rows sent to the dummy block.
        mtp.write_token_maps(
            gpu_view=self.gpu_view,
            rows=mtp.row_ids[:n],
            positions=positions,
            block_table=mtp.active_block_table,
            padded_token_count=padded,
        )

        # MHA read metadata: query_length=1 per request, kv_length = P_r + 1 (write-then-attend).
        query_lengths, kv_len = mtp.stage_decode_lengths()
        mtp.write_mha_metadata(
            gpu_view=self.gpu_view,
            query_lengths=query_lengths,
            kv_lengths=kv_len,
            block_table=mtp.active_block_table,
            padded_request_count=padded,
        )

        # The staged buffers above are read only by these writes; the graph reads `gpu_view`
        # alone, whose addresses are fixed, so replaying against them is graph-safe.
        if mtp.graphed:
            # Graphed: route to the graph metadata and use the FIXED capture-time sequence-length
            # bound (baked into the flash-attn kernel launch at capture) rather than a per-step
            # `.item()` sync. `kv_len = position + 1 <= max_sequence_length`, so `max_seqlen` is a
            # safe upper bound; the actual per-request lengths come from the GPU cu_kv tensors.
            max_seqlen_k = self.graph_attn_metadata["mha_metadata"].max_seqlen
        else:
            # Eager: tight per-step max via a GPU->CPU sync, non-graph metadata, graphs disabled.
            max_seqlen_k = int(kv_len.max().item()) if n > 0 else 1
        self._mtp_activate_attn_metadata(
            graphed=mtp.graphed,
            padded_request_count=padded,
            max_seqlen_q=1,
            max_seqlen_k=max_seqlen_k,
            token_count=n,
            padded_token_count=padded,
        )

    def _mtp_setup_prefill_step(
        self,
        append_counts: Tensor,
        block_table_prefill: Tensor,
        padded_token_count: Optional[int] = None,
        padded_request_count: Optional[int] = None,
        request_start_positions: Optional[Tensor] = None,
        total: Optional[int] = None,
    ) -> None:
        """Populate token write maps + MHA metadata for a varlen roll-by-one MTP write forward.

        Each request writes `append_counts[r]` consecutive positions starting at
        `request_start_positions[r]`, or at 0 when that is None (a fresh prompt seed).

        `append_counts`/`block_table_prefill` are GPU tensors for the P requests in active-slice
        order. `total` is `append_counts.sum()`; pass it when the caller already has it, to skip
        a redundant device sync.
        """
        assert self.enable_mtp_kv_cache
        gv = self.gpu_view
        device = gv.token_to_block_idx.device
        num_prefill = append_counts.numel()
        if total is None:
            total = int(append_counts.sum().item())
        padded_total = total if padded_token_count is None else padded_token_count
        padded_p = num_prefill if padded_request_count is None else padded_request_count

        # Per-token request row and within-request index (0..count-1); then shift each request's
        # write positions by its start offset (0 for prompt seed; committed offset for refresh).
        # `output_size` is known on the host, so pass it and skip repeat_interleave's own sync.
        rows = torch.repeat_interleave(
            torch.arange(num_prefill, device=device), append_counts, output_size=total
        )
        seg_start = torch.cumsum(append_counts, 0) - append_counts
        positions = torch.arange(total, device=device) - seg_start[rows]
        if request_start_positions is not None:
            positions = positions + request_start_positions.to(device)[rows]

        # Keep writes out of blocks this request inherited. The two spans overlap whenever the
        # prefix skip covers fewer tokens than the matched blocks span -- a short chunk, a
        # non-block-aligned resume, the `>= 2` clamp, a short Mamba match, or memory-only mode.
        active_slice = slice(self.paused_request_count, self.total_request_count)
        inherited = (
            self.request_matched_prefix_blocks[active_slice][: block_table_prefill.shape[0]]
            .to(block_table_prefill.device, non_blocking=True)
            .to(torch.long)
        )
        self.mtp_metadata.write_token_maps(
            gpu_view=gv,
            rows=rows,
            positions=positions,
            block_table=block_table_prefill,
            padded_token_count=padded_total,
            inherited_blocks=inherited,
        )

        # MHA metadata: fresh causal prefill, so per-request kv_length == query_length and one
        # staged length buffer serves as both. `stage_prefill_lengths` also absorbs the padded
        # query rows, either into a trailing pad request or into the last real request.
        seq_lengths = self.mtp_metadata.stage_prefill_lengths(
            append_counts=append_counts, pad_tokens=padded_total - total
        )
        p = seq_lengths.numel()
        padded_p = max(padded_p, p)

        self.mtp_metadata.write_mha_metadata(
            gpu_view=gv,
            query_lengths=seq_lengths,
            kv_lengths=seq_lengths,
            block_table=block_table_prefill,
            padded_request_count=padded_p,
        )

        # Read the bound back off `seq_lengths` rather than off `append_counts`: either branch
        # above may have introduced a run longer than any single `append_counts` entry.
        max_seqlen = int(seq_lengths.max().item()) if p > 0 else 1
        self._mtp_activate_attn_metadata(
            graphed=False,
            padded_request_count=padded_p,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            token_count=total,
            padded_token_count=padded_total,
        )
        # COMMIT rather than DRAFT: per-request query lengths differ here, so the attention must
        # take the varlen path. On a pure-decode step `num_prefill_requests == 0` would otherwise
        # make `is_decode_only()` True and route to the decode kernel, whose uniform
        # `q.reshape(num_requests, tokens_per_request, ...)` cannot express ragged input.
        self.mtp_metadata.forward_mode = MTPForwardMode.COMMIT
