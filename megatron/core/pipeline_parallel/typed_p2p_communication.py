# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Typed P2P for ordinary and interleaved 1F1B.

Each invocation owns FIFO descriptors per virtual chunk, never activation graphs.
Backward buffers follow that chunk's oldest forward, including dynamic THD shapes.
Prepared per-microbatch plans post fixed control/data receives without descriptors.
Callers without a plan retain the dynamic header protocol for compatibility.
"""

import json
from collections import deque
from dataclasses import dataclass

import torch
import torch.distributed as dist

from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator, _p2p_ops
from megatron.core.pipeline_parallel.pipeline_payload import (
    PipelineGradientMessage,
    PipelineGradients,
    PipelinePayload,
    PipelinePayloadFactory,
    PipelinePayloadPlan,
    PipelinePayloadSpec,
    PipelineTensorSpec,
)
from megatron.core.transformer.state_boundary import TensorField
from megatron.core.utils import nvtx_decorator

# Dynamic callers first send one length, followed by an unbounded host descriptor.
# Prepared callers exchange plan fingerprints once and use fixed control records.
_HEADER_SIZE = 1
_DTYPES = (
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.float64,
    torch.int32,
    torch.int64,
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.complex64,
    torch.complex128,
)


def _pack_header(specs, metadata, microbatch, chunk_id=0, *, boundary_id="pipeline"):
    descriptor = PipelinePayloadSpec(tuple(specs), tuple(metadata), boundary_id)
    if not all(type(value) is int for value in (*metadata, microbatch, chunk_id)):
        raise ValueError("Invalid typed pipeline host metadata")
    if min(microbatch, chunk_id) < 0:
        raise ValueError("Invalid typed pipeline execution identity")
    fields = []
    for spec in specs:
        if spec.dtype not in _DTYPES:
            raise ValueError("Invalid typed pipeline tensor dtype")
        fields.append(
            (
                spec.name,
                spec.shape,
                _DTYPES.index(spec.dtype),
                spec.requires_grad,
                spec.layout,
                spec.present,
                spec.key,
            )
        )
    return list(
        json.dumps(
            (1, microbatch, chunk_id, boundary_id, metadata, fields, descriptor.fingerprint),
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    )


def _decode_header(values, microbatch, chunk_id=0):
    try:
        version, batch, chunk, boundary, metadata, fields, fingerprint = json.loads(bytes(values))
        if (version, batch, chunk) != (1, microbatch, chunk_id):
            raise ValueError("Typed pipeline microbatch mismatch or chunk mismatch")
        if any(type(value) is not int for value in (version, batch, chunk, fingerprint)):
            raise ValueError("Invalid typed pipeline execution identity")
        if any(type(field[2]) is not int or not 0 <= field[2] < len(_DTYPES) for field in fields):
            raise ValueError("Invalid typed pipeline tensor dtype")
        specs = tuple(
            PipelineTensorSpec(
                name, TensorField(key or name, tuple(shape), _DTYPES[dtype], layout, grad, present)
            )
            for name, shape, dtype, grad, layout, present, key in fields
        )
        descriptor = PipelinePayloadSpec(specs, tuple(metadata), boundary)
        if descriptor.fingerprint != fingerprint:
            raise ValueError("Typed pipeline boundary schema fingerprint mismatch")
        return descriptor
    except (TypeError, IndexError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Invalid typed pipeline descriptor") from error


def _unpack_header(values, microbatch, chunk_id=0):
    descriptor = _decode_header(values, microbatch, chunk_id)
    return descriptor.tensor_specs, descriptor.metadata


@dataclass(frozen=True)
class _MessageRecord:
    descriptor: PipelinePayloadSpec
    microbatch: int
    chunk_id: int

    def identity(self):
        return (self.microbatch, self.chunk_id, self.descriptor.fingerprint)


class _P2PWork:
    """Keep one direction's wire buffers alive until all its requests are waited."""

    def __init__(self, requests, buffers, pending, progress=None):
        self._requests = requests
        self._buffers = buffers
        self._pending = pending
        self._progress = progress
        self.validation = None

    def wait(self) -> None:
        """Wait every field once and release the detached communication buffers."""
        if self._progress is not None:
            self._progress()
            self._progress = None
        for request in self._requests:
            request.wait()
        if self.validation is not None:
            tensor, expected = self.validation
            if tuple(tensor.tolist()) != expected:
                raise ValueError("Pipeline forward microbatch/chunk/boundary mismatch")
            self.validation = None
        self._requests.clear()
        self._buffers.clear()
        self._pending.pop(self, None)


class _ForwardReceive:
    """A prefetched header whose variable-sized data buffers are not allocated yet."""

    def __init__(self, communicator, header, header_work, chunk_id, microbatch):
        self._communicator = communicator
        self._header = header
        self._header_work = header_work
        self._chunk_id = chunk_id
        self._microbatch = microbatch
        self._data_work = None
        self._payload = None
        self._pending = communicator._pending
        self._pending[self] = None

    def post_data(self) -> None:
        """Post all data receives before another message can use this peer's FIFO."""
        if self._header is None:
            return
        self._header_work.wait()
        communicator = self._communicator
        length = int(self._header.item())
        if length <= 0:
            raise ValueError("Invalid typed pipeline descriptor length")
        self._header = None
        header = torch.empty(length, dtype=torch.int64, device=communicator.device)
        communicator._transfer(recv_prev=[header])
        self._payload, tensors = communicator._make_received_payload(
            header, self._chunk_id, self._microbatch
        )
        self._data_work = communicator._transfer(recv_prev=tensors, overlap=True)["recv_prev"]
        self._header = self._header_work = self._communicator = None
        if communicator._forward_header is self:
            communicator._forward_header = None

    def wait(self) -> None:
        """Complete the header and data receive before the model consumes this input."""
        self.post_data()
        if self._data_work is not None:
            self._data_work.wait()
            self._data_work = None
        self._pending.pop(self, None)

    def take_payload(self) -> PipelinePayload:
        """Transfer ownership to the schedule's activation queue exactly once."""
        self.wait()
        if self._payload is None:
            raise RuntimeError("The prefetched pipeline payload has already been consumed")
        payload, self._payload = self._payload, None
        return payload


class TypedP2PCommunicator(P2PCommunicator):
    """Exchange ordered tensor fields; the model adapter reconstructs received payloads."""

    def __init__(
        self,
        pp_group: dist.ProcessGroup,
        config: ModelParallelConfig,
        make_payload: PipelinePayloadFactory | list[PipelinePayloadFactory],
        *,
        forward_only: bool = False,
        device: torch.device | None = None,
        payload_plans: list[PipelinePayloadPlan] | None = None,
    ) -> None:
        super().__init__(pp_group, config)
        if config.use_ring_exchange_p2p:
            raise ValueError("Typed P2P does not support ring_exchange")
        self.make_payloads = make_payload if isinstance(make_payload, list) else [make_payload]
        num_chunks = config.virtual_pipeline_model_parallel_size or 1
        if config.overlap_p2p_comm and (num_chunks < 2 or config.batch_p2p_comm):
            raise ValueError("Typed P2P overlap requires VPP and batch_p2p_comm=False")
        if len(self.make_payloads) != num_chunks or not all(map(callable, self.make_payloads)):
            raise ValueError("Typed P2P requires a payload factory for every virtual chunk")
        self.forward_only = forward_only
        self.device = (
            torch.device("cuda", torch.cuda.current_device()) if device is None else device
        )
        self._sent: list[deque[_MessageRecord]] = [deque() for _ in range(num_chunks)]
        self._received: list[deque[_MessageRecord]] = [deque() for _ in range(num_chunks)]
        self._send_microbatch = [0] * num_chunks
        self._recv_microbatch = [0] * num_chunks
        self._pending: dict[_P2PWork | _ForwardReceive, None] = {}
        self._forward_header: _ForwardReceive | None = None
        if payload_plans is not None and (
            len(payload_plans) != num_chunks
            or not all(isinstance(plan, PipelinePayloadPlan) for plan in payload_plans)
        ):
            raise ValueError("Typed P2P requires a payload plan for every virtual chunk")
        self.payload_plans = payload_plans
        if payload_plans is not None:
            self._validate_plan_peers()

    def _validate_plan_peers(self):
        # Real groups are initialized before construction. The guard also allows
        # local transport test doubles to exercise request/buffer ownership.
        if not dist.is_initialized():
            return
        local = tuple(
            (
                tuple(None if d is None else d.fingerprint for d in plan.incoming),
                tuple(None if d is None else d.fingerprint for d in plan.outgoing),
            )
            for plan in self.payload_plans
        )
        plans = [None] * self.pp_group.size()
        dist.all_gather_object(plans, local, group=self.pp_group)
        count = len(self.make_payloads)
        if any(len(plan) != count for plan in plans):
            raise ValueError("Pipeline peers disagree on their chunk count")
        ordered = [plans[rank][chunk] for chunk in range(count) for rank in range(len(plans))]
        for left, right in zip(ordered, ordered[1:]):
            if left[1] != right[0]:
                raise ValueError("Pipeline peer schemas do not match")

    def _transfer(
        self, *, send_next=None, send_prev=None, recv_prev=None, recv_next=None, overlap=False
    ):
        directions = [
            ("send_next", send_next, self.next_rank, dist.isend),
            ("send_prev", send_prev, self.prev_rank, dist.isend),
            ("recv_prev", recv_prev, self.prev_rank, dist.irecv),
            ("recv_next", recv_next, self.next_rank, dist.irecv),
        ]
        # Forward precedes backward in each peer's stream, including PP=2 where
        # prev and next are the same rank. Group handles by direction, not peer.
        if self.current_stage % 2:
            directions = directions[2:] + directions[:2]
        ops, owners, handles = [], [], {}
        for name, tensors, peer, operation in directions:
            if tensors is None:
                continue
            buffers = [
                tensor.detach().contiguous() if operation is dist.isend else tensor
                for tensor in tensors
                if tensor is not None and tensor.numel()
            ]
            progress = self._post_forward_data if buffers and operation is dist.isend else None
            handle = _P2PWork([], buffers, self._pending, progress)
            handles[name] = handle
            self._pending[handle] = None
            for tensor in buffers:
                if self.config.batch_p2p_comm:
                    ops.append(dist.P2POp(operation, tensor, peer, self.pp_group))
                    owners.append(handle)
                else:
                    p2p_tensors = dict.fromkeys(
                        (
                            "tensor_send_next",
                            "tensor_send_prev",
                            "tensor_recv_prev",
                            "tensor_recv_next",
                        )
                    )
                    p2p_tensors[f"tensor_{name}"] = tensor
                    # Reuse Megatron's PP=2 communicator split: prev and next
                    # share a peer, but independent directions must make progress.
                    request = _p2p_ops(
                        **p2p_tensors,
                        group=self.pp_group,
                        prev_pipeline_rank=self.prev_rank,
                        next_pipeline_rank=self.next_rank,
                    )[name]
                    handle._requests.append(request)
        if ops:
            for handle, request in zip(owners, dist.batch_isend_irecv(ops)):
                handle._requests.append(request)
        if overlap:
            return handles
        for handle in handles.values():
            handle.wait()
        if ops and self.config.batch_p2p_sync and self.device.type == "cuda":
            torch.cuda.synchronize()

    def _gradient_tensors(self, gradients, specs):
        specs = tuple(spec for spec in specs if spec.present)
        if len(gradients) != len(specs):
            raise ValueError("Pipeline gradients must match the incoming field slots")
        tensors = []
        for spec, grad in zip(specs, gradients):
            if not spec.requires_grad:
                if grad is not None:
                    raise ValueError(f"Unexpected gradient for {spec.name}")
                continue
            if grad is None:
                grad = torch.zeros(spec.shape, dtype=spec.dtype, device=self.device)
            if (
                tuple(grad.shape) != spec.shape
                or grad.dtype != spec.dtype
                or grad.device != self.device
            ):
                raise ValueError(f"Invalid gradient for {spec.name}")
            tensors.append(grad)
        return tensors

    def _make_received_payload(self, header, chunk_id, microbatch):
        descriptor = _decode_header(header.tolist(), microbatch, chunk_id)
        return self._allocate_payload(descriptor, chunk_id, microbatch)

    def _allocate_payload(self, descriptor, chunk_id, microbatch=0):
        specs = descriptor.tensor_specs
        tensors = tuple(
            torch.empty(
                spec.shape, dtype=spec.dtype, device=self.device, requires_grad=spec.requires_grad
            )
            for spec in specs
            if spec.present
        )
        payload = self.make_payloads[chunk_id](tensors, descriptor)
        if payload.descriptor != descriptor:
            raise ValueError("Received schema or gradient slots disagree with the model payload")
        if not self.forward_only:
            self._received[chunk_id].append(_MessageRecord(descriptor, microbatch, chunk_id))
        return payload, tensors

    def _planned_payload(self, chunk_id, microbatch, *, outgoing):
        plan = self.payload_plans[chunk_id]
        descriptors = plan.outgoing if outgoing else plan.incoming
        if not 0 <= microbatch < len(descriptors) or descriptors[microbatch] is None:
            raise ValueError(
                f"Missing pipeline payload plan for chunk {chunk_id}, microbatch {microbatch}"
            )
        return descriptors[microbatch]

    def _validate_planned_output(self, output, chunk_id, microbatch):
        expected = self._planned_payload(chunk_id, microbatch, outgoing=True)
        actual = output.tensor_specs
        if (
            output.metadata != expected.metadata
            or output.boundary_id != expected.boundary_id
            or len(actual) != len(expected.tensor_specs)
        ):
            raise ValueError("Outgoing pipeline payload disagrees with the prepared metadata")
        for spec, planned in zip(actual, expected.tensor_specs):
            if spec != planned:
                raise ValueError(
                    f"Pipeline chunk {chunk_id}, microbatch {microbatch}, field {spec.name}: "
                    f"actual {spec} disagrees with prepared {planned}"
                )

    def _post_forward_data(self):
        # Both neighbors can be retiring old sends while a future input is
        # prefetched. Post its data before waiting so neither stalls the other.
        if self._forward_header is not None:
            self._forward_header.post_data()

    def resolve_forward(
        self, received: PipelinePayload | _ForwardReceive | None
    ) -> PipelinePayload | None:
        """Resolve a prefetched input before storing the concrete payload for backward."""
        return received.take_payload() if isinstance(received, _ForwardReceive) else received

    @nvtx_decorator()
    def _release_sent_output(self, output: PipelinePayload | None) -> None:
        """Keep autograd roots while Work handles own any unfinished wire transfers."""
        if output is not None and getattr(self.config, "deallocate_pipeline_outputs", False):
            output.release_output()

    def _exchange(
        self,
        *,
        output=None,
        gradients=None,
        recv_forward=False,
        recv_backward=False,
        forward_send_chunk_id=0,
        backward_send_chunk_id=0,
        forward_recv_chunk_id=0,
        backward_recv_chunk_id=0,
        overlap=False,
    ):
        if overlap and (len(self.make_payloads) < 2 or self.config.batch_p2p_comm):
            raise ValueError("Typed P2P overlap requires VPP and batch_p2p_comm=False")
        operations = [
            name
            for active, name in (
                (output is not None, "forward-send"),
                (gradients is not None, "backward-send"),
                (recv_forward, "forward-recv"),
                (recv_backward, "backward-recv"),
            )
            if active
        ]
        timer = (
            self.config.timers("-".join(operations), log_level=2)
            if self.config.timers is not None
            else None
        )
        if timer is not None:
            timer.start()
        try:
            forward_sends = backward_sends = None
            if output is not None:
                if not isinstance(output, PipelinePayload) or output.device != self.device:
                    raise ValueError("Expected a pipeline payload on the communication device")
                descriptor = output.descriptor
                descriptor.schema.validate(output.tensors)
                microbatch = self._send_microbatch[forward_send_chunk_id]
                destination_chunk = forward_send_chunk_id + int(self.is_pp_last_stage)
                record = _MessageRecord(descriptor, microbatch, destination_chunk)
                if self.payload_plans is not None:
                    self._validate_planned_output(output, forward_send_chunk_id, microbatch)
                    control = torch.tensor(record.identity(), dtype=torch.int64, device=self.device)
                    forward_sends = [control, *output.tensors]
                else:
                    values = _pack_header(
                        descriptor.tensor_specs,
                        descriptor.metadata,
                        microbatch,
                        destination_chunk,
                        boundary_id=descriptor.boundary_id,
                    )
                    length = torch.tensor([len(values)], dtype=torch.int64, device=self.device)
                    header = torch.tensor(values, dtype=torch.int64, device=self.device)
                    forward_sends = [length, header, *output.tensors]
                self._send_microbatch[forward_send_chunk_id] += 1
                if not self.forward_only:
                    self._sent[forward_send_chunk_id].append(record)

            if gradients is not None:
                record = self._received[backward_send_chunk_id].popleft()
                if isinstance(gradients, PipelineGradientMessage):
                    gradients = gradients.resolve()
                indices = record.descriptor.schema.grad_tensor_indices
                active = tuple(int(gradients[i] is not None) for i in indices)
                control = torch.tensor(
                    (*record.identity(), *active), dtype=torch.int64, device=self.device
                )
                backward_sends = [
                    control,
                    *self._gradient_tensors(gradients, record.descriptor.tensor_specs),
                ]

            # Reserve all fields of a previous dynamic forward before another
            # receive can advance the same peer's ordered stream.
            if self._forward_header is not None and (
                recv_forward or (recv_backward and self.prev_rank == self.next_rank)
            ):
                self._post_forward_data()

            backward_message, backward_receives = None, None
            if recv_backward:
                record = self._sent[backward_recv_chunk_id].popleft()
                fields = record.descriptor.schema.packed_fields
                indices = record.descriptor.schema.grad_tensor_indices
                tensors = tuple(
                    (
                        torch.empty(field.shape, dtype=field.dtype, device=self.device)
                        if field.differentiable
                        else None
                    )
                    for field in fields
                )
                identity = record.identity()
                control = torch.empty(
                    len(identity) + len(indices), dtype=torch.int64, device=self.device
                )
                backward_message = PipelineGradientMessage(tensors, control, identity, indices)
                backward_receives = [control, *(tensors[i] for i in indices)]

            received, forward_receives, forward_identity = None, None, None
            microbatch = None
            if recv_forward:
                microbatch = self._recv_microbatch[forward_recv_chunk_id]
                self._recv_microbatch[forward_recv_chunk_id] += 1
                if self.payload_plans is None:
                    header = torch.empty(_HEADER_SIZE, dtype=torch.int64, device=self.device)
                    forward_receives = [header]
                else:
                    descriptor = self._planned_payload(
                        forward_recv_chunk_id, microbatch, outgoing=False
                    )
                    received, tensors = self._allocate_payload(
                        descriptor, forward_recv_chunk_id, microbatch
                    )
                    forward_identity = _MessageRecord(
                        descriptor, microbatch, forward_recv_chunk_id
                    ).identity()
                    control = torch.empty(
                        len(forward_identity), dtype=torch.int64, device=self.device
                    )
                    forward_receives = [control, *tensors]

            # PP=2 can use the same peer for both directions. A dynamic header
            # must be decoded and all its data receives posted before gradients
            # from that peer, otherwise a gradient receive could consume header data.
            defer_backward = (
                recv_forward
                and recv_backward
                and self.payload_plans is None
                and self.prev_rank == self.next_rank
            )
            handles = self._transfer(
                send_next=forward_sends,
                send_prev=backward_sends,
                recv_prev=forward_receives,
                recv_next=None if defer_backward else backward_receives,
                overlap=True,
            )
            if recv_forward:
                if self.payload_plans is None:
                    received = _ForwardReceive(
                        self, header, handles["recv_prev"], forward_recv_chunk_id, microbatch
                    )
                    self._forward_header = received
                    handles["recv_prev"] = received
                else:
                    handles["recv_prev"].validation = (forward_receives[0], forward_identity)
            if defer_backward:
                received.post_data()
                handles.update(self._transfer(recv_next=backward_receives, overlap=True))
            self._release_sent_output(output)
            if overlap:
                return received, backward_message, handles
            if isinstance(received, _ForwardReceive):
                received = received.take_payload()
            for handle in handles.values():
                handle.wait()
            if (
                self.config.batch_p2p_comm
                and self.config.batch_p2p_sync
                and self.device.type == "cuda"
            ):
                torch.cuda.synchronize()
            return received, backward_message
        finally:
            if timer is not None:
                timer.stop()

    def recv_forward(self, tensor_shapes: object, is_first_stage: bool) -> PipelinePayload | None:
        """Receive the next forward's header and tensors."""
        if not is_first_stage:
            return self._exchange(recv_forward=True)[0]

    def send_forward(self, output_tensors: PipelinePayload, is_last_stage: bool) -> None:
        """Send a forward and save its specs for backward."""
        if not is_last_stage:
            self._exchange(output=output_tensors)

    def send_forward_recv_backward(
        self, output_tensors: PipelinePayload, tensor_shapes: object, is_last_stage: bool
    ) -> PipelineGradientMessage | None:
        """Send this forward while receiving the oldest pending forward's gradients."""
        if not is_last_stage:
            return self._exchange(output=output_tensors, recv_backward=True)[1]

    def recv_backward(
        self, tensor_shapes: object, is_last_stage: bool, *, recv_chunk_id: int = 0
    ) -> PipelineGradientMessage | None:
        """Receive the oldest pending forward's gradients during cooldown."""
        if not is_last_stage:
            return self._exchange(recv_backward=True, backward_recv_chunk_id=recv_chunk_id)[1]

    def send_backward(self, input_tensor_grads: PipelineGradients, is_first_stage: bool) -> None:
        """Return gradients in the incoming forward's field order."""
        if not is_first_stage:
            self._exchange(gradients=input_tensor_grads)

    def send_backward_recv_forward(
        self, input_tensor_grads: PipelineGradients, tensor_shapes: object, is_first_stage: bool
    ) -> PipelinePayload | None:
        """Return old gradients while receiving a new forward."""
        if not is_first_stage:
            return self._exchange(gradients=input_tensor_grads, recv_forward=True)[0]

    def send_forward_recv_forward(
        self,
        output_tensor: PipelinePayload | None,
        recv_prev: bool,
        tensor_shape: object,
        overlap_p2p_comm: bool = False,
        *,
        send_chunk_id: int = 0,
        recv_chunk_id: int = 0,
    ) -> (
        PipelinePayload
        | None
        | tuple[PipelinePayload | _ForwardReceive | None, dict[str, _P2PWork | _ForwardReceive]]
    ):
        """Exchange forwards; resolve prefetched inputs before model execution."""
        result = self._exchange(
            output=output_tensor,
            recv_forward=recv_prev,
            forward_send_chunk_id=send_chunk_id,
            forward_recv_chunk_id=recv_chunk_id,
            overlap=overlap_p2p_comm,
        )
        return (result[0], result[2]) if overlap_p2p_comm else result[0]

    def send_backward_recv_backward(
        self,
        input_tensor_grad: PipelineGradients | None,
        recv_next: bool,
        tensor_shape: object,
        overlap_p2p_comm: bool = False,
        *,
        send_chunk_id: int = 0,
        recv_chunk_id: int = 0,
    ) -> (
        PipelineGradientMessage | None | tuple[PipelineGradientMessage | None, dict[str, _P2PWork]]
    ):
        """Exchange gradients; wait the returned receive handle before backward."""
        result = self._exchange(
            gradients=input_tensor_grad,
            recv_backward=recv_next,
            backward_send_chunk_id=send_chunk_id,
            backward_recv_chunk_id=recv_chunk_id,
            overlap=overlap_p2p_comm,
        )
        return (result[1], result[2]) if overlap_p2p_comm else result[1]

    def send_forward_backward_recv_forward_backward(
        self,
        output_tensor: PipelinePayload | None,
        input_tensor_grad: PipelineGradients | None,
        recv_prev: bool,
        recv_next: bool,
        tensor_shape: object,
        *,
        forward_send_chunk_id: int = 0,
        backward_send_chunk_id: int = 0,
        forward_recv_chunk_id: int = 0,
        backward_recv_chunk_id: int = 0,
    ) -> tuple[PipelinePayload | None, PipelineGradientMessage | None]:
        """Exchange all four directions using independent chunk/microbatch descriptors."""
        return self._exchange(
            output=output_tensor,
            gradients=input_tensor_grad,
            recv_forward=recv_prev,
            recv_backward=recv_next,
            forward_send_chunk_id=forward_send_chunk_id,
            backward_send_chunk_id=backward_send_chunk_id,
            forward_recv_chunk_id=forward_recv_chunk_id,
            backward_recv_chunk_id=backward_recv_chunk_id,
        )

    def finish(self) -> None:
        """Drain outstanding P2P and check that every microbatch completed backward."""
        self._post_forward_data()
        for handle in list(self._pending):
            handle.wait()
        if any(self._sent) or any(self._received):
            raise RuntimeError("Typed pipeline schedule left unfinished microbatches")
        if self.payload_plans is not None:
            for chunk, plan in enumerate(self.payload_plans):
                if self._send_microbatch[chunk] != sum(
                    s is not None for s in plan.outgoing
                ) or self._recv_microbatch[chunk] != sum(s is not None for s in plan.incoming):
                    raise RuntimeError("Typed pipeline schedule did not consume its payload plan")
