# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Model-independent typed pipeline stages: receive, restore, run, export and plan."""

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Protocol, Sequence

import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.pipeline_payload import (
    PipelineDataIterator,
    PipelinePayloadPlan,
    PipelinePayloadSpec,
    TensorStatePayload,
)
from megatron.core.transformer.state_boundary import TensorField
from megatron.core.utils import get_attr_wrapped_model

# Host integers that describe one microbatch (e.g. its batch shape).
PipelineMetadata = tuple[int, ...]


class PipelineStateCodec(Protocol):
    """Model-specific half of a typed stage; boundaries are global layer indices.

    Every stage derives a boundary's fields from the same metadata, so peers agree
    on each message before any activation is computed.
    """

    name: str

    def describe(
        self, input_ids: torch.Tensor, packed_seq_params: PackedSeqParams | None
    ) -> PipelineMetadata:
        """Describe a first-stage microbatch without reading device values."""
        ...

    def fields(self, boundary: int, metadata: PipelineMetadata) -> tuple[TensorField, ...]:
        """Return the hidden field, then the state fields live before layer ``boundary``."""
        ...

    def initial_state(
        self,
        input_ids: torch.Tensor,
        packed_seq_params: PackedSeqParams | None,
        metadata: PipelineMetadata,
    ) -> dict[str, torch.Tensor | None]:
        """Build the first stage's state (e.g. tokens, cumulative sequence lengths)."""
        ...

    def packed_seq_params(
        self, state: dict[str, torch.Tensor | None], metadata: PipelineMetadata
    ) -> PackedSeqParams | None:
        """Rebuild packed-sequence metadata on a later stage from received state."""
        ...


@dataclass
class PipelineStageState:
    """One microbatch's state on a stage; layer-side consumers may replace ``values``."""

    metadata: PipelineMetadata = ()
    values: dict[str, torch.Tensor | None] = field(default_factory=dict)
    packed_seq_params: PackedSeqParams | None = None


class PipelineStage:
    """Typed transport for the layers ``[start, end)`` of one physical or virtual stage.

    Holds construction metadata and at most one received payload, which the next
    forward consumes; activations otherwise belong to ``PipelineStageState``.
    """

    def __init__(
        self,
        codec: PipelineStateCodec,
        start: int,
        end: int,
        *,
        pre_process: bool,
        post_process: bool,
    ) -> None:
        self.codec, self.start, self.end = codec, start, end
        self.pre_process, self.post_process = pre_process, post_process
        self._incoming: tuple[PipelinePayloadSpec, dict] | None = None

    def spec(self, boundary: int, metadata: PipelineMetadata) -> PipelinePayloadSpec:
        """Describe the message crossing ``boundary`` for one microbatch."""
        metadata = tuple(metadata)
        return PipelinePayloadSpec(
            self.codec.fields(boundary, metadata), metadata, f"{self.codec.name}:{boundary}"
        )

    def plan(self, metadata: Sequence[PipelineMetadata]) -> PipelinePayloadPlan:
        """Return this stage's incoming/outgoing descriptors in microbatch order."""
        return PipelinePayloadPlan(
            tuple(None if self.pre_process else self.spec(self.start, m) for m in metadata),
            tuple(None if self.post_process else self.spec(self.end, m) for m in metadata),
        )

    def receive(self, input_tensor: Any) -> Any:
        """``set_input_tensor`` hook: retain a payload's state and return its hidden tensor."""
        self._incoming = None
        if not isinstance(input_tensor, TensorStatePayload):
            return input_tensor
        hidden, values = input_tensor.restore()
        self._incoming = (input_tensor.spec, values)
        return hidden

    def enter(
        self, input_ids: torch.Tensor | None, packed_seq_params: PackedSeqParams | None
    ) -> PipelineStageState:
        """Restore the received state, or build it from the batch on the first stage."""
        incoming, self._incoming = self._incoming, None
        if incoming is None:
            if not self.pre_process:
                raise ValueError("A later pipeline stage requires a typed state payload")
            metadata = tuple(self.codec.describe(input_ids, packed_seq_params))
            values = self.codec.initial_state(input_ids, packed_seq_params, metadata)
            return PipelineStageState(metadata, dict(values), packed_seq_params)
        spec, values = incoming
        if spec.boundary_id != f"{self.codec.name}:{self.start}":
            raise ValueError(f"Received payload {spec.boundary_id} belongs to another boundary")
        packed_seq_params = self.codec.packed_seq_params(values, spec.metadata)
        return PipelineStageState(spec.metadata, values, packed_seq_params)

    def exit(self, hidden: torch.Tensor, state: PipelineStageState) -> TensorStatePayload:
        """Publish the hidden activation and the state live after this stage."""
        hidden_field, *fields = self.codec.fields(self.end, state.metadata)
        return TensorStatePayload.from_state(
            hidden,
            state.values,
            fields,
            hidden_field=hidden_field,
            boundary_id=f"{self.codec.name}:{self.end}",
            metadata=state.metadata,
        )


def prepare_stage_inputs(
    data_iterator: Iterator[Any],
    model: torch.nn.Module,
    num_microbatches: int,
    *,
    forward_only: bool,
    describe: Callable[[Any], PipelineMetadata],
) -> PipelineDataIterator:
    """A ``forward_step_func.prepare_pipeline_inputs`` hook for models with a stage.

    Bind ``describe`` (batch -> metadata) with ``functools.partial``. Every stage,
    including middle stages, must see each microbatch's metadata in forward order.
    """
    del forward_only  # Plans are identical with and without backward.
    stage = get_attr_wrapped_model(model, "pipeline_stage")
    batches = [next(data_iterator) for _ in range(num_microbatches)]
    return PipelineDataIterator(batches, stage.plan([describe(batch) for batch in batches]))
