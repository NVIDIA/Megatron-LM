# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""GPU-side input preparation helpers for dynamic inference snapshots."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from .decode_metadata_kernels import prepare_decode_metadata
from .gpu_input_state import GpuSampledTokenState

if TYPE_CHECKING:
    from megatron.core.inference.engines.dynamic_step import (
        DynamicStepContextSnapshot,
        StepInputPlan,
    )


class GpuInputPreparer:
    """Scatters GPU-resident sampled tokens into a prepared snapshot."""

    def __init__(
        self,
        *,
        stream: torch.cuda.Stream,
        block_size_tokens: int,
        debug_enabled: bool = False,
    ):
        self.stream = stream
        self.block_size_tokens = int(block_size_tokens)
        self.debug_enabled = bool(debug_enabled)
        self._last_input_ready_event: Optional[torch.cuda.Event] = None

    def prepare(
        self,
        snapshot: "DynamicStepContextSnapshot",
        input_plan: "StepInputPlan",
        previous_sample_state: Optional[GpuSampledTokenState],
    ) -> torch.cuda.Event:
        """Prepare GPU inputs for ``snapshot`` and return an input-ready event."""
        if snapshot.snapshot_slot_id != input_plan.snapshot_slot_id:
            raise RuntimeError(
                f"Input plan targets snapshot slot {input_plan.snapshot_slot_id}, "
                f"but snapshot slot is {snapshot.snapshot_slot_id}"
            )
        if snapshot.gpu_view is None:
            raise RuntimeError("GPU input preparation requires a snapshot gpu_view")
        bound_slot_id = getattr(snapshot.gpu_view, "current_snapshot_slot_id", None)
        if bound_slot_id != snapshot.snapshot_slot_id:
            raise RuntimeError(
                f"Snapshot gpu_view is bound to slot {bound_slot_id}, "
                f"not slot {snapshot.snapshot_slot_id}"
            )

        decode_count = len(input_plan.decode_input_destination_indices)
        if decode_count and previous_sample_state is None:
            raise RuntimeError("Decode input preparation requires previous sampled-token state")

        token_buffer = snapshot.gpu_view.token_to_input_ids
        debug_expected_metadata = None
        input_ready_event = torch.cuda.Event()
        with torch.cuda.stream(self.stream):
            if self.debug_enabled and decode_count:
                debug_expected_metadata = self._capture_decode_metadata(snapshot, input_plan)
            if decode_count:
                self.stream.wait_event(previous_sample_state.sample_gpu_ready_event)
                destinations = torch.tensor(
                    input_plan.decode_input_destination_indices,
                    dtype=torch.long,
                    device=token_buffer.device,
                )
                token_buffer.index_copy_(
                    0, destinations, previous_sample_state.sampled_tokens[:decode_count]
                )
                if input_plan.speculative_width > 0:
                    if previous_sample_state.sampled_mtp_tokens is None:
                        raise RuntimeError(
                            "Speculative input preparation requested without sampled MTP state"
                        )
                    if input_plan.speculative_width > previous_sample_state.num_speculative_tokens:
                        raise RuntimeError(
                            "Input plan speculative width exceeds sampled-token state width"
                        )
                    for depth in range(input_plan.speculative_width):
                        token_buffer.index_copy_(
                            0,
                            destinations + depth + 1,
                            previous_sample_state.sampled_mtp_tokens[depth, :decode_count],
                        )
                prepare_decode_metadata(
                    snapshot.gpu_view,
                    decode_request_slots=input_plan.decode_request_slots,
                    decode_input_destination_indices=input_plan.decode_input_destination_indices,
                    speculative_width=input_plan.speculative_width,
                    block_size_tokens=self.block_size_tokens,
                )
            input_ready_event.record(self.stream)

        self._last_input_ready_event = input_ready_event
        if self.debug_enabled and input_plan.debug_expected_input_ids is not None:
            self._debug_compare(
                snapshot,
                input_plan,
                input_ready_event,
                debug_expected_metadata=debug_expected_metadata,
            )

        return input_ready_event

    def _decode_token_indices(self, input_plan: "StepInputPlan") -> torch.Tensor:
        """Return CPU token indices covered by decode input preparation."""
        compare_indices = []
        for base_idx in input_plan.decode_input_destination_indices:
            compare_indices.extend(
                int(base_idx) + offset for offset in range(input_plan.speculative_width + 1)
            )
        return torch.tensor(compare_indices, dtype=torch.long, device="cpu")

    def _capture_decode_metadata(
        self, snapshot: "DynamicStepContextSnapshot", input_plan: "StepInputPlan"
    ) -> dict:
        """Capture CPU-prepared GPU metadata before the GPU decode helper overwrites it."""
        gpu_view = snapshot.gpu_view
        token_indices_cpu = self._decode_token_indices(input_plan)
        token_indices_gpu = token_indices_cpu.to(gpu_view.token_to_pos_ids.device)
        request_slots_cpu = torch.tensor(
            input_plan.decode_request_slots, dtype=torch.long, device="cpu"
        )
        request_slots_gpu = request_slots_cpu.to(gpu_view.token_to_pos_ids.device)
        decode_count = len(input_plan.decode_request_slots)
        return {
            "token_indices_cpu": token_indices_cpu,
            "request_slots_cpu": request_slots_cpu,
            "token_to_pos_ids": gpu_view.token_to_pos_ids.index_select(
                0, token_indices_gpu
            ).clone(),
            "token_to_request_idx": gpu_view.token_to_request_idx.index_select(
                0, token_indices_gpu
            ).clone(),
            "token_to_position_in_request": gpu_view.token_to_position_in_request.index_select(
                0, token_indices_gpu
            ).clone(),
            "token_to_local_position_within_kv_block": (
                gpu_view.token_to_local_position_within_kv_block.index_select(
                    0, token_indices_gpu
                ).clone()
            ),
            "token_to_block_idx": gpu_view.token_to_block_idx.index_select(
                0, token_indices_gpu
            ).clone(),
            "request_query_lengths": gpu_view.request_query_lengths.index_select(
                0, request_slots_gpu
            ).clone(),
            "mha_query_lengths": gpu_view.mha_query_lengths.index_select(
                0, request_slots_gpu
            ).clone(),
            "mha_kv_seq_lengths": gpu_view.mha_kv_seq_lengths.index_select(
                0, request_slots_gpu
            ).clone(),
            "mha_cu_query_seq_lengths": gpu_view.mha_cu_query_seq_lengths[
                : decode_count + 1
            ].clone(),
            "mha_cu_kv_seq_lengths": gpu_view.mha_cu_kv_seq_lengths[
                : decode_count + 1
            ].clone(),
        }

    def _debug_compare(
        self,
        snapshot: "DynamicStepContextSnapshot",
        input_plan: "StepInputPlan",
        input_ready_event: torch.cuda.Event,
        *,
        debug_expected_metadata: Optional[dict] = None,
    ) -> None:
        """Synchronize in debug mode and compare scatter destinations."""
        input_ready_event.synchronize()
        if not input_plan.decode_input_destination_indices:
            return

        index_cpu = self._decode_token_indices(input_plan)
        index_gpu = index_cpu.to(snapshot.gpu_view.token_to_input_ids.device, non_blocking=True)
        actual = snapshot.gpu_view.token_to_input_ids.index_select(0, index_gpu).cpu()
        expected_source = input_plan.debug_expected_input_ids
        if torch.is_tensor(expected_source):
            expected = expected_source.to(device="cpu").index_select(0, index_cpu)
        else:
            expected = torch.tensor(
                [expected_source[int(idx)] for idx in index_cpu.tolist()],
                dtype=actual.dtype,
                device="cpu",
        )
        if not torch.equal(actual, expected):
            raise RuntimeError("GPU input scatter destinations diverged from CPU input IDs")
        if debug_expected_metadata is not None:
            self._debug_compare_metadata(snapshot, debug_expected_metadata)

    def _debug_compare_metadata(
        self, snapshot: "DynamicStepContextSnapshot", expected_metadata: dict
    ) -> None:
        """Compare GPU-prepared decode metadata against the CPU-prepared H2D snapshot."""
        gpu_view = snapshot.gpu_view
        token_indices_gpu = expected_metadata["token_indices_cpu"].to(
            gpu_view.token_to_pos_ids.device
        )
        request_slots_gpu = expected_metadata["request_slots_cpu"].to(
            gpu_view.token_to_pos_ids.device
        )
        comparisons = {
            "token_to_pos_ids": gpu_view.token_to_pos_ids.index_select(0, token_indices_gpu),
            "token_to_request_idx": gpu_view.token_to_request_idx.index_select(
                0, token_indices_gpu
            ),
            "token_to_position_in_request": gpu_view.token_to_position_in_request.index_select(
                0, token_indices_gpu
            ),
            "token_to_local_position_within_kv_block": (
                gpu_view.token_to_local_position_within_kv_block.index_select(
                    0, token_indices_gpu
                )
            ),
            "token_to_block_idx": gpu_view.token_to_block_idx.index_select(
                0, token_indices_gpu
            ),
            "request_query_lengths": gpu_view.request_query_lengths.index_select(
                0, request_slots_gpu
            ),
            "mha_query_lengths": gpu_view.mha_query_lengths.index_select(0, request_slots_gpu),
            "mha_kv_seq_lengths": gpu_view.mha_kv_seq_lengths.index_select(
                0, request_slots_gpu
            ),
            "mha_cu_query_seq_lengths": gpu_view.mha_cu_query_seq_lengths[
                : expected_metadata["mha_cu_query_seq_lengths"].numel()
            ],
            "mha_cu_kv_seq_lengths": gpu_view.mha_cu_kv_seq_lengths[
                : expected_metadata["mha_cu_kv_seq_lengths"].numel()
            ],
        }
        for name, actual_gpu in comparisons.items():
            actual = actual_gpu.cpu()
            expected = expected_metadata[name].cpu()
            if not torch.equal(actual, expected):
                raise RuntimeError(f"GPU-prepared decode metadata diverged for {name}")
