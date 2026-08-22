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
    2. Project the packed input (`in_proj`).
    3. Split the packed batch into a decode partition (1 token per request,
       placed first) and a prefill partition (variable length, placed after).
       The kernels cannot mix the two, so they run independently.
    4. Run the decode and prefill kernels on their respective partitions.
    5. Merge the two partitions back into packed token order.
    6. Apply the output projection (`out_proj`).

Only the kernels in step 4 differ between variants. This mixin owns the shared
control flow (steps 1-3, 5, 6 and the orchestration) and delegates the
variant-specific work to two hooks, `ssm_decode` and `ssm_prefill`. New
linear-attention variants should subclass this mixin and implement those two
hooks rather than re-deriving the decode/prefill bookkeeping.

Both hooks are given the `DynamicInferenceContext` directly and read whatever
per-step metadata they need from `context.mamba_metadata` /
`context.mamba_slot_allocator` themselves; there is deliberately no
intermediate "unpack the metadata into a long argument list" layer.

Speculative decoding is supported by the shared orchestration: the decode path
reshapes tokens into `[batch, seq_len, d]`, fetches intermediate state buffers
from the context, and passes them to `ssm_decode`. Variants that do not yet
support speculative decoding should assert `seq_len == 1` inside their
`ssm_decode` implementation.

Chunked prefill and prefix caching are handled entirely inside `ssm_prefill`
via `context.mamba_metadata` and `context.mamba_slot_allocator`; the mixin
orchestration is unaware of them.

Note: static-batching ("legacy") inference is intentionally *not* part of this
interface. Concrete mixers keep any static/eager inference path separate so
it does not pollute the dynamic decode/prefill hooks defined here.
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional, Sequence, Tuple

import torch

from megatron.core.inference.contexts import DynamicInferenceContext
from megatron.core.inference.contexts.attention_context.triton.tensor_ops import (
    tensor_get_slice_after,
    tensor_merge,
)
from megatron.core.utils import is_using_quantization_scales


class SSMChunking(NamedTuple):
    """Chunk-related facts shared by every SSM layer in a stack."""

    chunk_size: int
    """The mixer's configured chunk size."""

    inference_chunk_size: int
    """The chunk length the dynamic-inference prefill kernels actually run at."""

    num_householder: int
    """Householder copies for Gated Delta Product layers; 0 for other mixers."""


def ssm_chunking(layer_type_list: List[str], layers: Sequence) -> Optional[SSMChunking]:
    """Returns the chunking every SSM layer in a stack shares, or None.

    None means the stack holds no recurrent layer, which happens on a pipeline
    stage made up entirely of attention and MLP layers.

    The stack is assumed homogeneous: one mixer type, one chunking. A mixed
    stack would need a per-mixer alignment quantum and per-mixer chunk
    descriptors, and nothing downstream models that, so it is rejected here
    rather than silently taking the first layer's answer for every layer.

    Args:
        layer_type_list: Per-layer symbols, positionally matching `layers`. See
            `megatron/core/models/hybrid/hybrid_layer_allocation.py`.
        layers: The stack's layers.

    Returns:
        The shared `SSMChunking`, or None if no layer is recurrent.
    """
    from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols

    chunking = None
    first_layer_idx = None
    for layer_idx, (layer_type, layer) in enumerate(zip(layer_type_list, layers)):
        # Mamba-family mixers (including Gated Delta Product) hang off `.mixer`;
        # Gated Delta Net registers its recurrent mixer in the attention slot.
        if layer_type == Symbols.MAMBA:
            mixer = getattr(layer, 'mixer', None)
        elif layer_type == Symbols.GDN:
            mixer = getattr(layer, 'self_attention', None)
        else:
            continue
        if mixer is None:
            continue

        layer_chunking = SSMChunking(
            chunk_size=mixer.chunk_size,
            # The chunk length the inference kernels actually run at, which is
            # not always `chunk_size`: the forked Gated Delta Product prefill
            # kernels chunk at a fixed 64. Falls back to chunk_size for any
            # mixer predating the property.
            inference_chunk_size=getattr(mixer, 'ssm_inference_chunk_size', mixer.chunk_size),
            # Gated Delta Product layers register as Mamba layers but carry a
            # Householder count, which sizes their (separate) chunk descriptors.
            num_householder=getattr(mixer, 'num_householder', 0) or 0,
        )
        if chunking is None:
            chunking, first_layer_idx = layer_chunking, layer_idx
        else:
            assert layer_chunking == chunking, (
                f"every SSM layer must share one chunking; layer {first_layer_idx} has "
                f"{chunking} but layer {layer_idx} has {layer_chunking}"
            )
    return chunking


class SSMDynamicInferenceMixin:
    """Mixin providing the shared decode/prefill orchestration for the dynamic
    inference engine. Concrete mixers implement the two `ssm_*` hooks below."""

    # ------------------------------------------------------------------
    # Hooks implemented by concrete mixers.
    # ------------------------------------------------------------------
    def ssm_decode(
        self,
        zxBCdt: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        batch_indices: torch.Tensor,
        intermediate_conv_state: torch.Tensor = None,
        intermediate_ssm_state: torch.Tensor = None,
    ) -> torch.Tensor:
        """Run the single-token-per-request decode kernels.

        Args:
            zxBCdt: `[decode_req_count, seq_len, proj_dim]` projected decode tokens,
                where `seq_len = 1 + num_speculative_tokens`.
            conv_state: `[num_slots, conv_channels, d_conv]` conv state cache.
            ssm_state: `[num_slots, *ssm_shape]` SSM state cache.
            batch_indices: `[decode_req_count]` slot index per decode request
                (`-1` marks padding slots).
            intermediate_conv_state: Optional buffer for storing conv states at
                intermediate sequence steps (speculative decoding).
            intermediate_ssm_state: Optional buffer for storing SSM states at
                intermediate sequence steps (speculative decoding).

        Returns `[decode_req_count, seq_len, d_inner]`; updates state in place.
        Variants that do not yet support speculative decoding should assert
        `seq_len == 1` inside their implementation.
        """
        raise NotImplementedError

    def ssm_prefill(
        self,
        zxBCdt: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        context: DynamicInferenceContext,
    ) -> torch.Tensor:
        """Run the variable-length prefill kernels for all prefill requests.

        The implementation reads its varlen metadata (`cu_seqlens`,
        `batch_indices_prefill`, `seq_idx`, chunk boundaries, intermediate
        extraction buffers, etc.) directly from `context.mamba_metadata` and
        `context.mamba_slot_allocator` and processes every prefill request in
        one varlen call, writing the resulting final states back into the caches.

        Returns `[prefill_token_count, 1, d_inner]`; updates state in place.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared orchestration.
    # ------------------------------------------------------------------
    def ssm_dynamic_inference(
        self, hidden_states: torch.Tensor, context: DynamicInferenceContext
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute one dynamic inference step for a linear-attention mixer.

        Separates decode and prefill requests, runs them through the
        variant-specific kernels independently, and merges the results back
        into packed token order.
        """
        # Grab standard states.
        conv_state, ssm_state = context.mamba_states_cache(self.layer_number - self.pp_layer_offset)

        # Fetch intermediate state buffers for speculative decoding.
        # These are pre-allocated output buffers; existing data is overwritten.
        int_conv_state = None
        int_ssm_state = None
        if context.num_speculative_tokens > 0:
            int_conv_state, int_ssm_state = context.mamba_states_cache(
                self.layer_number - self.pp_layer_offset, intermediate=True
            )

        padded_dims = context.padded_batch_dimensions
        token_count = padded_dims.token_count
        decode_req_count = padded_dims.decode_req_count
        prefill_req_count = padded_dims.prefill_req_count

        # Input projection over the full packed batch.
        zxBCdt, _ = self.in_proj(hidden_states)

        y_decode = None
        y_prefill = None

        # --- Decode partition (placed first in the packed batch) ---------
        if decode_req_count > 0:
            seq_len = 1 + context.num_speculative_tokens
            decode_token_count = decode_req_count * seq_len
            zxBCdt_decode = zxBCdt[:decode_token_count] if prefill_req_count > 0 else zxBCdt
            # Reshape from [N*S, 1, d] to [N, S, d] for the decode kernels.
            zxBCdt_decode = zxBCdt_decode.squeeze(1).view(decode_req_count, seq_len, -1)
            # ReplaySSM (Mamba-2 speculative decoding): hand the hook the context so
            # it can read the ring buffers / cursors itself. Passed only when the
            # feature is enabled, so hooks that do not support it are unaffected.
            replay_kwargs = {}
            if getattr(context, "mamba_replay_ssm", False):
                replay_kwargs = {"replay_context": context}
            y_decode = self.ssm_decode(
                zxBCdt_decode,
                conv_state,
                ssm_state,
                batch_indices=context.mamba_metadata.batch_indices_decode,
                intermediate_conv_state=int_conv_state,
                intermediate_ssm_state=int_ssm_state,
                **replay_kwargs,
            )
            # Flatten back to [N*S, 1, d] to match the merge logic.
            y_decode = y_decode.view(decode_token_count, 1, -1)

        # --- Prefill partition -------------------------------------------
        if prefill_req_count > 0:
            if decode_req_count > 0:
                # Mixed batch: gather the prefill tokens out of the packed tensor.
                zxBCdt_prefill = torch.empty_like(zxBCdt)
                tensor_get_slice_after(
                    zxBCdt,
                    zxBCdt_prefill,
                    context.mamba_metadata.device_decode_prefill,
                    check_bounds=False,
                )
            else:
                zxBCdt_prefill = zxBCdt
            y_prefill = self.ssm_prefill(zxBCdt_prefill, conv_state, ssm_state, context)

        # --- Merge back into packed token order --------------------------
        if y_decode is not None and y_prefill is not None:
            y = torch.empty(
                [token_count, 1, y_prefill.shape[-1]],
                dtype=y_prefill.dtype,
                device=y_prefill.device,
            )
            tensor_merge(
                y_decode, y_prefill, context.mamba_metadata.device_decode_prefill, output_tensor=y
            )
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
