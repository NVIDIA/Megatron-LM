# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GPU input preparation for the async-overlap inference pipeline (v3 plan
§2.7).

Owns the scatter that populates the snapshot's ``token_to_input_ids`` from
the previous step's GPU-resident sampled tokens. Stream-ordered against
``prev_sample_ready_event`` and ``snapshot.metadata_ready_event``; records
``input_ready_event`` so the forward stream can stream-wait on it.

This module is wired in commit 17. With ``enable_async_overlap=False`` the
preparer is not consumed (the legacy CPU write into ``token_to_input_ids``
inside ``update_requests`` remains authoritative); commits 18+ enable the
GPU path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from megatron.core.inference.engines.async_pipeline_types import StepInputPlan


@dataclass
class PrevSampleState:
    """Per-step handles consumed by ``GpuInputPreparer.prepare``.

    The previous step's GPU-resident sampled tokens (already copied into
    ``prev_sampled_token_ids`` on the gpu_bookkeeping stream by commit 16)
    plus the events the preparer must stream-wait on.
    """

    prev_sampled_token_ids: torch.Tensor
    sample_done_event: Optional[torch.cuda.Event]
    prev_sample_ready_event: Optional[torch.cuda.Event]


class GpuInputPreparer:
    """Owns the GPU input-id scatter for the async-overlap pipeline.

    Stream layout (v3 plan §2.6): all work runs on
    ``gpu_bookkeeping_stream``. The stream waits on both
    ``prev_sample_state.prev_sample_ready_event`` (so the D2D into
    ``prev_sampled_token_ids`` has completed) and
    ``snapshot.metadata_ready_event`` (so the destination indices in
    ``snapshot._buf`` are valid). Records ``input_ready_event`` for the
    forward stream to stream-wait on.
    """

    def __init__(self, gpu_bookkeeping_stream: torch.cuda.Stream) -> None:
        self._stream = gpu_bookkeeping_stream

    def prepare(
        self,
        snapshot,
        input_plan: "StepInputPlan",
        prev_sample_state: PrevSampleState,
        is_first_step: bool = False,
    ) -> torch.cuda.Event:
        """Run the input prep for one step. Returns ``input_ready_event``.

        With ``is_first_step=True`` the scatter is bypassed (no prior step
        produced a sample) and a no-op event is recorded so consumers can
        stream-wait unconditionally.
        """
        # Always record an event on the gpu_bookkeeping stream so the
        # forward stream's stream-wait protocol is uniform across the
        # first-step and steady-state paths.
        if is_first_step:
            event = torch.cuda.Event()
            event.record(self._stream)
            return event

        if prev_sample_state.prev_sample_ready_event is not None:
            self._stream.wait_event(prev_sample_state.prev_sample_ready_event)
        metadata_event = getattr(snapshot, "metadata_ready_event", None)
        if metadata_event is not None:
            self._stream.wait_event(metadata_event)

        decode_indices = input_plan.decode_token_destination_indices
        with torch.cuda.stream(self._stream):
            if decode_indices:
                n = len(decode_indices)
                indices_gpu = torch.tensor(
                    list(decode_indices),
                    dtype=torch.long,
                    device=prev_sample_state.prev_sampled_token_ids.device,
                )
                snapshot.token_to_input_ids.scatter_(
                    0,
                    indices_gpu,
                    prev_sample_state.prev_sampled_token_ids[:n],
                )
            event = torch.cuda.Event()
            event.record(self._stream)
        return event
