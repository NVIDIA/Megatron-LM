# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared dynamic-batching inference scaffolding for linear-attention mixers.

A growing family of mixers in Megatron behave like "linear attention" / SSM
recurrences for inference purposes: they carry a small per-request recurrent
state (a short-convolution state plus a matrix-valued SSM state) instead of a
growing KV cache. Mamba was the first; Gated Delta Net / Gated Delta Product
(GDP) and friends are the same shape of computation with different kernels.

All of these variants share an *identical* request-level control flow for the
dynamic inference engine:

    1. Fetch this layer's (conv_state, ssm_state) slabs from the context.
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

MVP scope: this path deliberately does not yet cover speculative decoding,
Mamba prefix caching, chunked prefill, or CUDA-graph capture. Those are layered
on top of the same hooks incrementally; ``MambaMixer`` keeps its own fully
optimized override for now. See ``GatedDeltaProductMixer`` for the first
adopter.
"""

from __future__ import annotations

from typing import Tuple

import torch

from megatron.core.inference.contexts import DynamicInferenceContext
from megatron.core.inference.contexts.attention_context.triton.tensor_ops import (
    tensor_get_slice_after,
    tensor_merge,
)


class SSMDynamicInferenceMixin:
    """Mixin providing the shared decode/prefill orchestration for the dynamic
    inference engine. Concrete mixers implement the ``_ssm_*`` hooks below."""

    # ------------------------------------------------------------------
    # Hooks implemented by concrete mixers.
    # ------------------------------------------------------------------
    def _ssm_in_projection(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project packed ``[token_count, 1, d_model]`` input to the mixer's
        internal projection layout ``[token_count, 1, proj_dim]``."""
        raise NotImplementedError

    def _ssm_decode(
        self,
        proj: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        batch_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Run the single-token-per-request decode kernels.

        Args:
            proj: ``[decode_token_count, 1, proj_dim]`` projected decode tokens.
            conv_state: ``[num_slots, conv_channels, d_conv]`` conv state cache.
            ssm_state: ``[num_slots, *ssm_shape]`` SSM state cache.
            batch_indices: ``[decode_req_count]`` slot index per decode request
                (``-1`` marks padding slots).

        Returns ``[decode_token_count, 1, d_inner]``; updates state in place.
        """
        raise NotImplementedError

    def _ssm_prefill(
        self,
        proj: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
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

    def _ssm_out_projection(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the output projection to the merged ``[token_count, 1, d_inner]``
        activation, returning ``(out, out_bias)``."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared orchestration.
    # ------------------------------------------------------------------
    def ssm_dynamic_inference(
        self, hidden_states: torch.Tensor, context: DynamicInferenceContext
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute one dynamic inference step for a linear-attention mixer.

        Mirrors the decode/prefill split that ``MambaMixer._dynamic_inference``
        performs, but is kernel-agnostic: the actual recurrences happen inside
        the ``_ssm_decode`` / ``_ssm_prefill`` hooks.
        """
        # The dynamic context lays the conv/ssm state slabs out per *mamba*
        # layer index; GDP-style layers register as Mamba layers, so the same
        # accessor and layer_map apply.
        conv_state, ssm_state = context.ssm_states_cache(
            self.layer_number - self.pp_layer_offset
        )

        padded_dims = context.padded_batch_dimensions
        token_count = padded_dims.token_count
        decode_req_count = padded_dims.decode_req_count
        prefill_req_count = padded_dims.prefill_req_count
        metadata = context.ssm_metadata

        # Input projection over the full packed batch.
        proj = self._ssm_in_projection(hidden_states)

        y_decode = None
        y_prefill = None

        # --- Decode partition (placed first in the packed batch) ---------
        if decode_req_count > 0:
            # MVP: exactly one token per decode request (no speculative tokens).
            assert context.num_speculative_tokens == 0, (
                "Linear-attention dynamic inference MVP does not support "
                "speculative decoding yet."
            )
            decode_token_count = decode_req_count
            proj_decode = proj[:decode_token_count] if prefill_req_count > 0 else proj
            y_decode = self._ssm_decode(
                proj_decode, conv_state, ssm_state, metadata.batch_indices_decode
            )

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
            y_prefill = self._ssm_prefill(proj_prefill, conv_state, ssm_state, context)

        # --- Merge back into packed token order --------------------------
        if y_decode is not None and y_prefill is not None:
            y = torch.empty(
                [token_count, 1, y_prefill.shape[-1]],
                dtype=y_prefill.dtype,
                device=y_prefill.device,
            )
            tensor_merge(
                y_decode, y_prefill, metadata.device_decode_prefill, output_tensor=y
            )
        elif y_decode is not None:
            y = y_decode
        elif y_prefill is not None:
            y = y_prefill
        else:
            raise RuntimeError(
                "Dynamic inference called with 0 decode and 0 prefill requests"
            )

        return self._ssm_out_projection(y)
