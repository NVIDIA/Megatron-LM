# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""GPU-side input preparation helpers for dynamic inference snapshots."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from .gpu_input_state import GpuSampledTokenState

if TYPE_CHECKING:
    from megatron.core.inference.engines.dynamic_step import (
        DynamicStepContextSnapshot,
        StepInputPlan,
    )


class GpuInputPreparer:
    """Scatters GPU-resident sampled tokens into a prepared snapshot."""

    def __init__(self, *, stream: torch.cuda.Stream, debug_enabled: bool = False):
        self.stream = stream
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
        input_ready_event = torch.cuda.Event()
        with torch.cuda.stream(self.stream):
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
            input_ready_event.record(self.stream)

        self._last_input_ready_event = input_ready_event
        if self.debug_enabled and input_plan.debug_expected_input_ids is not None:
            self._debug_compare(snapshot, input_plan, input_ready_event)

        return input_ready_event

    def _debug_compare(
        self,
        snapshot: "DynamicStepContextSnapshot",
        input_plan: "StepInputPlan",
        input_ready_event: torch.cuda.Event,
    ) -> None:
        """Synchronize in debug mode and compare scatter destinations."""
        input_ready_event.synchronize()
        if not input_plan.decode_input_destination_indices:
            return

        compare_indices = []
        for base_idx in input_plan.decode_input_destination_indices:
            compare_indices.extend(
                int(base_idx) + offset for offset in range(input_plan.speculative_width + 1)
            )

        index_cpu = torch.tensor(compare_indices, dtype=torch.long, device="cpu")
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
