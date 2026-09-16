# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Multi-token-prediction (MTP) speculative decoding for dynamic inference.

`MTPInferenceMixin` holds the MTP half of `TextGenerationController`: the serial draft loop
that produces speculative tokens, and the dummy forwards that keep idle expert-parallel ranks
in lockstep with them. It is a mixin rather than a standalone helper because these paths read
a large amount of controller state (sampling buffers, the wrapped model, SP/PP topology); the
split is for readability, and mixing in preserves the previous behaviour exactly.

The only state the mixin writes is its own: the sampling/draft buffers allocated by
`_init_mtp_sampling_tensors`. Everything else it touches belongs to the inference context or
the model.
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

        if base_position is None:
            # Legacy scheduling derives positions from post-rewind CPU state.
            cuda_device = torch.cuda.current_device()
            adjusted_offsets = context.request_kv_length_offsets[active_slice].to(
                cuda_device, non_blocking=True
            )
            processed_tokens = context.request_query_lengths[active_slice].to(
                cuda_device, non_blocking=True
            )
            base_position = (adjusted_offsets + processed_tokens).to(torch.int64)

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

        for depth in range(self.num_mtp_depths):
            nvtx_range_push(f"mtp-spec-decoding/depth-{depth}")

            token_ids_buf[0, :active_request_count] = next_token_ids
            position_ids_buf[0, :active_request_count] = base_position + depth

            mtp_logits_2d = None
            if has_mtp:
                nvtx_range_push(f"mtp-spec-decoding/depth-{depth}/forward")
                mtp_depth = None if unwrapped_model.mtp.mtp_use_repeated_layer else depth
                current_hidden, mtp_logits = unwrapped_model.compute_mtp_single_step(
                    hidden_states=current_hidden,
                    next_token_ids=token_ids_buf,
                    position_ids=position_ids_buf,
                    depth=mtp_depth,
                    eager=not context.using_cuda_graph_this_step(),
                    cache_key=(
                        ("mtp", padded_count, mtp_depth)
                        if context.using_cuda_graph_this_step()
                        else None
                    ),
                )
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
                    eager=not context.using_cuda_graph_this_step(),
                    cache_key=(
                        ("mtp", padded_count, mtp_depth)
                        if context.using_cuda_graph_this_step()
                        else None
                    ),
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
