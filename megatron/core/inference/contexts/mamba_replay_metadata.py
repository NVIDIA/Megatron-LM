# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""ReplaySSM state for Mamba-2 speculative decoding.

ReplaySSM (https://tridao.me/blog/2026/replayssm/) keeps each slot's live SSM state as a
lazily-updated checkpoint plus a per-(layer, slot) circular ring buffer of the SSM inputs
cached since that checkpoint (post-conv `x|B` at conv-state precision, raw fp32 `dt`).
Verification outputs are computed from the checkpoint and the ring, the checkpoint is
re-materialized only on (early-)flush steps, and speculative rollback is a device-side move of
the ring cursors. All of it is persistent GPU state that the kernels update in place, so unlike
the per-step Mamba metadata nothing here rides the bookkeeping H2D.

`DynamicInferenceContext` holds a `MambaReplayMetadata` only when `mamba_replay_ssm` is enabled
with speculative decoding; otherwise `MambaReplayMetadata.create` returns None and no buffer or
bookkeeping is spent.
"""

from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor

from megatron.core.inference.config import MambaInferenceStateConfig
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from megatron.core.ssm.ops.mamba2.mamba_ssm_replay import commit_replayssm_spec

# dt drives the cumulative decay replay, so its cache stays fp32 regardless of state dtype.
_DT_CACHE_DTYPE = torch.float32


def _validate_mamba2_model(state_config: Optional[MambaInferenceStateConfig]) -> None:
    """Raise unless the model's recurrent layers are Mamba-2.

    The ReplaySSM kernels implement only the Mamba-2 recurrence. Gated DeltaNet has its own
    layer symbol; Gated Delta Product mixers sit in Mamba ("M") layers but report a nonzero
    Householder count. (A PP stage holding no SSM layer reports a count of 0, but it also never
    runs an SSM decode.)
    """
    layer_types = state_config.layer_type_list if state_config is not None else []
    if Symbols.GDN in layer_types:
        unsupported = "uses Gated DeltaNet layers"
    elif state_config is not None and state_config.gdp_num_householder > 0:
        unsupported = "uses Gated Delta Product layers"
    elif Symbols.MAMBA not in layer_types:
        unsupported = "has no Mamba layers"
    else:
        return
    raise ValueError(
        "mamba_replay_ssm requires a model whose recurrent layers are Mamba-2, "
        f"but this model {unsupported}."
    )


class MambaReplayMetadata:
    """ReplaySSM geometry, ring buffers, ring cursors, and per-step scratch.

    The geometry is fixed at construction because the context's memory budget needs it before
    `max_requests` is known; `allocate` and `deallocate` then follow the context's tensor
    lifecycle. Attribute names are part of the op-layer contract: `MambaMixer.ssm_decode` reads
    `window`, `flush_threshold`, `write_pos`, `origin`, `is_flush`, `qsl`, and `bc_pre` by
    attribute access, and the per-layer rings through `layer_caches`.
    """

    # Tensors whose contents must survive a KV-cache offload. `bc_pre` is per-step scratch and
    # `qsl` is constant, so neither needs saving.
    PERSISTENT_TENSORS = ("conv_cache", "dt_cache", "write_pos", "origin", "is_flush")

    def __init__(
        self,
        num_speculative_tokens: int,
        buffer_len: int,
        conv_states_shape: Tuple[int, ...],
        ssm_states_shape: Tuple[int, ...],
        conv_states_dtype: torch.dtype,
    ):
        """
        Args:
            num_speculative_tokens: Draft tokens per decode step (at least 1).
            buffer_len: Committed-history buffer length `B`, in tokens.
            conv_states_shape: Per-request conv state shape `(conv_dim, d_conv)`.
            ssm_states_shape: Per-request SSM state shape `(nheads, headdim, d_state)`.
            conv_states_dtype: Conv state dtype; the post-conv cache is stored at this precision.
        """
        # W: verification-window length (the guaranteed bonus token + the drafts).
        self.window = num_speculative_tokens + 1
        # B: committed-history buffer length.
        self.buffer_len = buffer_len
        if buffer_len < self.window:
            raise ValueError(
                f"mamba_replay_buffer_len ({buffer_len}) must be >= "
                f"1 + num_speculative_tokens ({self.window})"
            )
        # L: logical flush threshold; the physical ring is the next power of two so ring
        # indices wrap with a bit-mask.
        self.flush_threshold = buffer_len + self.window
        self.cache_buf_len = 1 << (self.flush_threshold - 1).bit_length()
        self.block_spec = 1 << (self.window - 1).bit_length()

        # The kernels need the TP-local group count, which the state shapes imply:
        # conv_dim = d_inner + 2 * ngroups * d_state, with d_inner = nheads * headdim.
        self.nheads, headdim, self.dstate = ssm_states_shape
        bc_dim = conv_states_shape[0] - self.nheads * headdim
        if bc_dim <= 0 or bc_dim % (2 * self.dstate) != 0:
            raise ValueError(
                f"mamba_replay_ssm cannot infer ngroups from conv_dim={conv_states_shape[0]}, "
                f"nheads={self.nheads}, headdim={headdim}, dstate={self.dstate}"
            )
        self.ngroups = bc_dim // (2 * self.dstate)
        if self.nheads % self.ngroups != 0:
            raise ValueError(
                f"mamba_replay_ssm: nheads ({self.nheads}) is not divisible by the inferred "
                f"ngroups ({self.ngroups})"
            )
        # The post-conv cache holds x|B only; C is read fresh from each step's conv output.
        self.cache_conv_dim = self.nheads * headdim + self.ngroups * self.dstate
        self.cache_dtype = conv_states_dtype

        self.max_requests = None
        self.deallocate()

    @staticmethod
    def create(
        enabled: bool,
        num_speculative_tokens: int,
        buffer_len: int,
        state_config: Optional[MambaInferenceStateConfig],
    ) -> Optional["MambaReplayMetadata"]:
        """The ReplaySSM state, or None when ReplaySSM is not in use.

        Args:
            enabled: Whether `mamba_replay_ssm` is requested.
            num_speculative_tokens: Draft tokens per decode step. ReplaySSM is inert without
                speculative decoding, so 0 returns None (after validating the model).
            buffer_len: Committed-history buffer length `B`, in tokens.
            state_config: The model's Mamba inference state config; None for models without
                recurrent layers.

        Raises:
            ValueError: If enabled for a model whose recurrent layers are not Mamba-2, or with
                `buffer_len < 1 + num_speculative_tokens`.
        """
        if not enabled:
            return None
        _validate_mamba2_model(state_config)
        if num_speculative_tokens == 0:
            return None
        return MambaReplayMetadata(
            num_speculative_tokens,
            buffer_len,
            state_config.conv_states_shape,
            state_config.ssm_states_shape,
            state_config.conv_states_dtype,
        )

    def ring_bytes_per_request(self, num_layers: int) -> int:
        """Bytes of ring buffers one request adds across `num_layers` Mamba layers."""
        post_conv_bytes = self.cache_conv_dim * self.cache_dtype.itemsize
        dt_bytes = self.nheads * _DT_CACHE_DTYPE.itemsize
        return self.cache_buf_len * (post_conv_bytes + dt_bytes) * num_layers

    def allocate(self, num_layers: int, max_requests: int, device: torch.device) -> None:
        """Reserve the ring buffers, cursors, and scratch.

        Args:
            num_layers: Number of Mamba layers on this rank.
            max_requests: Number of Mamba state slots.
            device: Device for the buffers.
        """
        self.max_requests = max_requests
        buf = self.cache_buf_len
        # Circular post-conv input cache (packed x|B).
        self.conv_cache = torch.zeros(
            (num_layers, max_requests, buf, self.cache_conv_dim),
            dtype=self.cache_dtype,
            device=device,
        )
        # Raw-dt cache; dt_bias and softplus are applied on read.
        self.dt_cache = torch.zeros(
            (num_layers, max_requests, self.nheads, buf), dtype=_DT_CACHE_DTYPE, device=device
        )
        # Slot-keyed ring cursors, shared by every Mamba layer (each layer commits the same
        # tokens). Advanced only by `commit`; cleared by `reset_slots`.
        self.write_pos = torch.zeros(max_requests, dtype=torch.int32, device=device)
        self.origin = torch.zeros(max_requests, dtype=torch.int32, device=device)
        self.is_flush = torch.zeros(max_requests, dtype=torch.int8, device=device)
        # Per-step scratch for the precomputed B.C inner products, at a fixed address for CUDA
        # graphs and rewritten by every layer within a step.
        self.bc_pre = torch.zeros(
            (max_requests, self.ngroups, buf, self.block_spec), dtype=torch.float32, device=device
        )
        # Fixed query_start_loc: every decode request occupies exactly `window` tokens in the
        # packed decode token stream.
        self.qsl = torch.arange(
            0, (max_requests + 1) * self.window, self.window, dtype=torch.int32, device=device
        )

    def deallocate(self) -> None:
        """Release the buffers, mirroring `allocate`.

        Used by the context's suspend path, which drops its tensors and rebuilds them from
        `initialize_all_tensors` on resume.
        """
        self.conv_cache = None
        self.dt_cache = None
        self.write_pos = None
        self.origin = None
        self.is_flush = None
        self.bc_pre = None
        self.qsl = None

    def layer_caches(self, mamba_layer_idx: int) -> Tuple[Tensor, Tensor]:
        """The `(post_conv_cache, dt_cache)` rings of one Mamba layer."""
        return self.conv_cache[mamba_layer_idx], self.dt_cache[mamba_layer_idx]

    def reset_slots(self, slots: Sequence[int]) -> None:
        """Empty the rings of the given state slots.

        Call whenever a slot's live SSM state becomes exact for its request (zeroed, restored
        from the prefix cache, or imported by a disaggregated handoff), so the kernels treat it
        as the checkpoint with no cached inputs.
        """
        if not slots:
            return
        idx = torch.tensor(slots, dtype=torch.long, device=self.write_pos.device)
        self.write_pos[idx] = 0
        self.origin[idx] = 0
        self.is_flush[idx] = 0

    def assert_rings_empty(self, slots: Sequence[int]) -> None:
        """Assert that the live SSM states of the given slots are exact.

        A slot's live state may only be read out as a full state (e.g. saved into the prefix
        cache) while its ring is empty; otherwise the ring must be folded in first.
        """
        if not slots:
            return
        idx = torch.tensor(slots, dtype=torch.long, device=self.write_pos.device)
        assert not bool((self.write_pos[idx] != 0).any().item()), (
            "ReplaySSM: reading out the live SSM state of a slot with a non-empty ring buffer; "
            "the state is stale and must be flushed first."
        )

    def commit(
        self, accepted_draft_counts: Tensor, prefill_status: Tensor, state_batch_indices: Tensor
    ) -> None:
        """Commit one speculative step: the entire ReplaySSM rollback.

        Advances each decode row's ring cursors past its committed tokens (the bonus token plus
        the accepted drafts) and precomputes the next step's flush flags, all on device.
        Rejected drafts stay behind the new write position and are overwritten by the next
        step's scatter, so no state is copied. Prefill rows are left untouched.

        Args:
            accepted_draft_counts: `(batch,)` accepted draft tokens per row.
            prefill_status: `(batch,)` 1 for prefill rows, 0 for decode rows.
            state_batch_indices: `(batch,)` Mamba state slot per row.
        """
        commit_replayssm_spec(
            write_pos=self.write_pos,
            post_conv_state_pos=self.origin,
            is_flush=self.is_flush,
            accepted_draft_counts=accepted_draft_counts,
            prefill_status=prefill_status,
            state_batch_indices=state_batch_indices,
            max_cache_len=self.flush_threshold,
            max_spec_len=self.window,
            cache_buf_len=self.cache_buf_len,
        )

    def warmup_commit(self) -> None:
        """Compile the commit kernel ahead of the first decode step.

        The commit runs eagerly after each decode step, outside the CUDA graphs, so its first
        Triton compile would otherwise land on the first real step. Every row is marked as
        prefill, so the call leaves the cursors untouched.
        """
        device = self.write_pos.device
        n = self.max_requests
        with torch.inference_mode():
            self.commit(
                accepted_draft_counts=torch.zeros(n, dtype=torch.int64, device=device),
                prefill_status=torch.ones(n, dtype=torch.int32, device=device),
                # int32 to match the real call (request_to_mamba_state_idx is int32); another
                # index dtype would compile a separate Triton specialization.
                state_batch_indices=torch.arange(n, dtype=torch.int32, device=device),
            )
        torch.cuda.synchronize()
