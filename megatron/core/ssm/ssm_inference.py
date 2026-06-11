# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared dynamic-batching inference scaffolding for linear-attention mixers.

A growing family of mixers in Megatron behave like "linear attention" / SSM
recurrences for inference purposes: they carry a small per-request recurrent
state (a short-convolution state plus a matrix-valued SSM state) instead of a
growing KV cache. Mamba was the first; Gated Delta Net / Gated Delta Product
(GDP) and friends are the same shape of computation with different kernels.

All of these variants share an *identical* request-level control flow for the
dynamic inference engine:

    1. Fetch this layer's (conv_state, recurrent_state) slabs from the context.
    2. Project the packed input.
    3. Split the packed batch into a decode partition (1 token per request,
       placed first) and a prefill partition (variable length, placed after).
       The kernels cannot mix the two, so they run independently.
    4. Merge the two partitions back into packed token order.
    5. Apply the output projection.

Only the kernels in steps 2-5 differ between variants. This mixin owns the
shared control flow (steps 1, 3, 4 and the orchestration) and delegates the
variant-specific work to a small set of hooks. New linear-attention variants
should subclass this mixin and implement the four ``_ssm_*`` hooks rather than
re-deriving the decode/prefill bookkeeping.

Speculative decoding is supported by the shared orchestration: the decode path
reshapes tokens into ``[batch, seq_len, d]``, fetches intermediate state buffers
from the context, and passes them to ``_ssm_decode``. Variants that do not yet
support speculative decoding should assert ``batch_size == 1 or seq_len == 1``
(or equivalent) inside their ``_ssm_decode`` implementation.

Chunked prefill and prefix caching are handled entirely inside ``_ssm_prefill``
via ``context.ssm_metadata`` and ``context.ssm_slot_allocator``; the mixin
orchestration is unaware of them.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from megatron.core.inference.contexts import DynamicInferenceContext
from megatron.core.inference.contexts.attention_context.triton.tensor_ops import (
    tensor_get_slice_after,
    tensor_merge,
)
from megatron.core.utils import is_using_quantization_scales


class SSMDynamicInferenceMixin:
    """Mixin providing the shared decode/prefill orchestration for the dynamic
    inference engine. Concrete mixers implement the ``_ssm_*`` hooks below."""

    # ------------------------------------------------------------------
    # Hooks implemented by concrete mixers.
    # ------------------------------------------------------------------
    def _ssm_decode(
        self,
        proj: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        batch_indices: torch.Tensor,
        intermediate_conv_state: Optional[torch.Tensor] = None,
        intermediate_recurrent_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the single-token-per-request decode kernels.

        Args:
            proj: ``[decode_req_count, seq_len, proj_dim]`` projected decode tokens,
                where ``seq_len = 1 + num_speculative_tokens``.
            conv_state: ``[num_slots, conv_channels, d_conv]`` conv state cache.
            recurrent_state: ``[num_slots, *ssm_shape]`` SSM state cache.
            batch_indices: ``[decode_req_count]`` slot index per decode request
                (``-1`` marks padding slots).
            intermediate_conv_state: Optional buffer for storing conv states at
                intermediate sequence steps (speculative decoding).
            intermediate_recurrent_state: Optional buffer for storing SSM states at
                intermediate sequence steps (speculative decoding).

        Returns ``[decode_req_count, seq_len, d_inner]``; updates state in place.
        Variants that do not yet support speculative decoding should assert
        ``seq_len == 1`` inside their implementation.
        """
        raise NotImplementedError

    def _ssm_prefill(
        self,
        proj: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        context: DynamicInferenceContext | None = None,
    ) -> torch.Tensor:
        """Run the variable-length prefill kernels for all prefill requests.

        When ``context`` is provided (dynamic inference), the implementation
        should extract metadata (``cu_seqlens``, ``batch_indices_prefill``,
        ``seq_idx``, etc.) from ``context.ssm_metadata`` and use them to
        process every prefill request in one varlen call, writing the resulting
        final states back into the caches.

        When ``context`` is ``None`` (static inference), no dynamic-batching
        metadata is available and the implementation should fall back to a
        standard single-sequence prefill.

        Returns ``[prefill_token_count, 1, d_inner]``; updates state in place.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared orchestration.
    # ------------------------------------------------------------------
    def ssm_dynamic_inference(
        self, hidden_states: torch.Tensor, context: DynamicInferenceContext
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute one dynamic inference step for a linear-attention mixer."""
        conv_state, recurrent_state = context.ssm_states_cache(
            self.layer_number - self.pp_layer_offset
        )

        # Fetch intermediate state buffers for speculative decoding.
        # These are pre-allocated output buffers; existing data is overwritten.
        int_conv_state = None
        int_recurrent_state = None
        if context.num_speculative_tokens > 0:
            int_conv_state, int_recurrent_state = context.ssm_states_cache(
                self.layer_number - self.pp_layer_offset, intermediate=True
            )

        padded_dims = context.padded_batch_dimensions
        token_count = padded_dims.token_count
        decode_req_count = padded_dims.decode_req_count
        prefill_req_count = padded_dims.prefill_req_count
        metadata = context.ssm_metadata

        # Input projection over the full packed batch.
        proj, _ = self.in_proj(hidden_states)

        y_decode = None
        y_prefill = None

        # --- Decode partition (placed first in the packed batch) ---------
        if decode_req_count > 0:
            seq_len = 1 + context.num_speculative_tokens
            decode_token_count = decode_req_count * seq_len
            proj_decode = proj[:decode_token_count] if prefill_req_count > 0 else proj
            # Reshape from [N*S, 1, d] to [N, S, d] for the decode kernels.
            proj_decode = proj_decode.squeeze(1).view(decode_req_count, seq_len, -1)
            y_decode = self._ssm_decode(
                proj_decode,
                conv_state,
                recurrent_state,
                metadata.batch_indices_decode,
                intermediate_conv_state=int_conv_state,
                intermediate_recurrent_state=int_recurrent_state,
            )
            # Flatten back to [N*S, 1, d] to match the merge logic.
            y_decode = y_decode.view(decode_token_count, 1, -1)

        # --- Prefill partition -------------------------------------------
        if prefill_req_count > 0:
            if decode_req_count > 0:
                # Mixed batch: gather the prefill tokens out of the packed tensor.
                proj_prefill = torch.empty_like(proj)
                tensor_get_slice_after(
                    proj, proj_prefill, metadata.device_decode_prefill, check_bounds=False
                )
            else:
                proj_prefill = proj
            y_prefill = self._ssm_prefill(proj_prefill, conv_state, recurrent_state, context)

        # --- Merge back into packed token order --------------------------
        if y_decode is not None and y_prefill is not None:
            y = torch.empty(
                [token_count, 1, y_prefill.shape[-1]],
                dtype=y_prefill.dtype,
                device=y_prefill.device,
            )
            tensor_merge(y_decode, y_prefill, metadata.device_decode_prefill, output_tensor=y)
        elif y_decode is not None:
            y = y_decode
        elif y_prefill is not None:
            y = y_prefill
        else:
            raise RuntimeError("Dynamic inference called with 0 decode and 0 prefill requests")

        # Zero padding positions to avoid corrupting quantization amax calculations.
        if is_using_quantization_scales(self.config):
            y[context.padding_slice] = 0.0

        return self.out_proj(y)
