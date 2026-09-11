# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Execute one recurrent operation using inference-context-owned state buffers.

The mixin handles projection, decode/prefill partitioning, state updates and
output merging. Global cache allocation, request scheduling and stack-wide
configuration remain with the inference context and model assembly.
"""

from typing import Tuple

import torch

from megatron.core.inference.contexts import DynamicInferenceContext
from megatron.core.inference.contexts.attention_context.triton.tensor_ops import (
    tensor_get_slice_after,
    tensor_merge,
)
from megatron.core.utils import is_using_quantization_scales


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
            if context.batch_invariant_mode:
                # Batch-invariant execution may include token-only rows to preserve
                # model-wide M alignment. Those rows do not represent requests and
                # must not be passed to the recurrent decode kernels.
                assert decode_token_count <= zxBCdt.shape[0], (
                    "Batch-invariant SSM metadata describes more decode tokens "
                    f"({decode_token_count}) than the input projection contains "
                    f"({zxBCdt.shape[0]})."
                )
                zxBCdt_decode = zxBCdt[:decode_token_count]
            else:
                zxBCdt_decode = zxBCdt[:decode_token_count] if prefill_req_count > 0 else zxBCdt
            # Reshape from [N*S, 1, d] to [N, S, d] for the decode kernels.
            zxBCdt_decode = zxBCdt_decode.squeeze(1).view(decode_req_count, seq_len, -1)
            y_decode = self.ssm_decode(
                zxBCdt_decode,
                conv_state,
                ssm_state,
                batch_indices=context.mamba_metadata.batch_indices_decode,
                intermediate_conv_state=int_conv_state,
                intermediate_ssm_state=int_ssm_state,
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

        if context.batch_invariant_mode:
            # Restore the projection's token-only padding before the output projection.
            # Its row count can be TP-local, unlike the context's global token count.
            padding_token_count = zxBCdt.shape[0] - y.shape[0]
            assert padding_token_count >= 0, (
                "Batch-invariant SSM produced more token rows "
                f"({y.shape[0]}) than the input projection contained ({zxBCdt.shape[0]})."
            )
            if padding_token_count > 0:
                y = torch.cat((y, y.new_zeros(padding_token_count, *y.shape[1:])), dim=0)

        # Zero padding positions to avoid corrupting quantization amax calculations.
        if is_using_quantization_scales(self.config):
            y[context.padding_slice] = 0.0

        return self.out_proj(y)
