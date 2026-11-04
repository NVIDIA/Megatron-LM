# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Prepared typed boundaries for eager non-interleaved pipeline training.

The schema, ownership and gradient activity protocol are extracted from PR 7224.
A fresh communicator and host plan belong to each schedule invocation.
"""

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist

from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator, _p2p_ops
from megatron.core.pipeline_parallel.pipeline_payload import (
    PipelineGradientMessage,
    PipelineGradients,
    PipelinePayload,
    PipelinePayloadFactory,
    PipelinePayloadPlan,
    PipelinePayloadSpec,
)
from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass(frozen=True)
class _MessageRecord:
    descriptor: PipelinePayloadSpec
    microbatch: int
    source_chunk_id: int
    destination_chunk_id: int

    def identity(self):
        return (
            self.microbatch,
            self.source_chunk_id,
            self.destination_chunk_id,
            self.descriptor.fingerprint,
        )


class _P2PWork:
    """Keep one direction's wire buffers alive until all its requests are waited."""

    def __init__(self, requests, buffers, pending):
        self._requests = requests
        self._buffers = buffers
        self._pending = pending
        self.validation = None

    def wait(self) -> None:
        """Wait every field once and release the detached communication buffers."""
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


class TypedP2PCommunicator(P2PCommunicator):
    """Transport a prepared field schema through the ordinary 1F1B schedule."""

    def __init__(
        self,
        pp_group: dist.ProcessGroup,
        config: TransformerConfig,
        make_payload: PipelinePayloadFactory | Sequence[PipelinePayloadFactory],
        *,
        payload_plan: PipelinePayloadPlan | Sequence[PipelinePayloadPlan],
        forward_only: bool = False,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(pp_group, config)
        if (
            config.overlap_p2p_comm
            or config.use_ring_exchange_p2p
            or config.tensor_model_parallel_size != 1
            or config.context_parallel_size != 1
            or config.recompute_granularity is not None
            or config.cuda_graph_impl != "none"
        ):
            raise ValueError(
                "Typed boundaries require eager PP without overlap, TP, CP, recompute or graphs"
            )
        self.make_payloads = (
            list(make_payload) if isinstance(make_payload, Sequence) else [make_payload]
        )
        self.payload_plans = (
            list(payload_plan) if isinstance(payload_plan, Sequence) else [payload_plan]
        )
        if len(self.make_payloads) != len(self.payload_plans):
            raise ValueError("Each virtual pipeline chunk requires a payload plan and factory")
        expected_chunks = config.virtual_pipeline_model_parallel_size or 1
        if len(self.make_payloads) != expected_chunks:
            raise ValueError(
                "Payload factories and plans must match virtual_pipeline_model_parallel_size"
            )
        self.forward_only = forward_only
        self.device = (
            torch.device("cuda", torch.cuda.current_device()) if device is None else device
        )
        self._sent = [deque() for _ in self.make_payloads]
        self._received = [deque() for _ in self.make_payloads]
        self._send_microbatch = [0 for _ in self.make_payloads]
        self._recv_microbatch = [0 for _ in self.make_payloads]
        self._pending: dict[_P2PWork, None] = {}
        self._validate_plan_peers()

    def _forward_destination_chunk_id(self, source_chunk_id: int) -> int:
        """Return the chunk which receives a forward payload from this rank."""
        if self.total_stages == 1 or self.current_stage != self.total_stages - 1:
            return source_chunk_id
        return (source_chunk_id + 1) % len(self.make_payloads)

    def _forward_source_chunk_id(self, destination_chunk_id: int) -> int:
        """Return the chunk which produced a forward payload received by this rank."""
        if self.total_stages == 1 or self.current_stage != 0:
            return destination_chunk_id
        return (destination_chunk_id - 1) % len(self.make_payloads)

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
        for rank in range(len(plans) - 1):
            for chunk in range(count):
                if plans[rank][chunk][1] != plans[rank + 1][chunk][0]:
                    raise ValueError("Pipeline peer schemas do not match")
        if len(plans) > 1:
            last_rank = len(plans) - 1
            for source_chunk in range(count):
                destination_chunk = (source_chunk + 1) % count
                if plans[last_rank][source_chunk][1] != plans[0][destination_chunk][0]:
                    raise ValueError("Pipeline peer schemas do not match at the VPP wrap")

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
            handle = _P2PWork([], buffers, self._pending)
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
            self._received[chunk_id].append(
                _MessageRecord(
                    descriptor, microbatch, self._forward_source_chunk_id(chunk_id), chunk_id
                )
            )
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

    def _exchange(
        self,
        *,
        output=None,
        gradients=None,
        recv_forward=False,
        recv_backward=False,
        send_forward_chunk_id=0,
        send_backward_chunk_id=0,
        recv_forward_chunk_id=0,
        recv_backward_chunk_id=0,
    ):
        forward_sends = backward_sends = forward_receives = backward_receives = None
        received = backward_message = None
        if output is not None:
            descriptor = output.descriptor
            descriptor.schema.validate(output.tensors)
            microbatch = self._send_microbatch[send_forward_chunk_id]
            self._validate_planned_output(output, send_forward_chunk_id, microbatch)
            record = _MessageRecord(
                descriptor,
                microbatch,
                send_forward_chunk_id,
                self._forward_destination_chunk_id(send_forward_chunk_id),
            )
            forward_sends = [
                torch.tensor(record.identity(), dtype=torch.int64, device=self.device),
                *output.tensors,
            ]
            self._send_microbatch[send_forward_chunk_id] += 1
            if not self.forward_only:
                self._sent[send_forward_chunk_id].append(record)
        if gradients is not None:
            if not self._received[send_backward_chunk_id]:
                raise RuntimeError(
                    f"No received payload is available for chunk {send_backward_chunk_id}"
                )
            record = self._received[send_backward_chunk_id].popleft()
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
        if recv_backward:
            if not self._sent[recv_backward_chunk_id]:
                raise RuntimeError(
                    f"No sent payload is available for chunk {recv_backward_chunk_id}"
                )
            record = self._sent[recv_backward_chunk_id].popleft()
            fields = record.descriptor.schema.packed_fields
            indices = record.descriptor.schema.grad_tensor_indices
            tensors = tuple(
                (
                    torch.empty(f.shape, dtype=f.dtype, device=self.device)
                    if f.differentiable
                    else None
                )
                for f in fields
            )
            identity = record.identity()
            control = torch.empty(
                len(identity) + len(indices), dtype=torch.int64, device=self.device
            )
            backward_message = PipelineGradientMessage(tensors, control, identity, indices)
            backward_receives = [control, *(tensors[i] for i in indices)]
        if recv_forward:
            microbatch = self._recv_microbatch[recv_forward_chunk_id]
            self._recv_microbatch[recv_forward_chunk_id] += 1
            descriptor = self._planned_payload(recv_forward_chunk_id, microbatch, outgoing=False)
            received, tensors = self._allocate_payload(
                descriptor, recv_forward_chunk_id, microbatch
            )
            identity = self._received[recv_forward_chunk_id][-1].identity()
            control = torch.empty(len(identity), dtype=torch.int64, device=self.device)
            forward_receives = [control, *tensors]
        handles = self._transfer(
            send_next=forward_sends,
            send_prev=backward_sends,
            recv_prev=forward_receives,
            recv_next=backward_receives,
            overlap=True,
        )
        if recv_forward:
            handles["recv_prev"].validation = (forward_receives[0], identity)
        for handle in handles.values():
            handle.wait()
        if output is not None and self.config.deallocate_pipeline_outputs:
            output.release_output()
        return received, backward_message

    def recv_forward(
        self, tensor_shapes: object, is_first_stage: bool, *, recv_chunk_id: int = 0
    ) -> PipelinePayload | None:
        """Receive the next forward's header and tensors."""
        if not is_first_stage:
            return self._exchange(recv_forward=True, recv_forward_chunk_id=recv_chunk_id)[0]

    def send_forward(
        self, output_tensors: PipelinePayload, is_last_stage: bool, *, send_chunk_id: int = 0
    ) -> None:
        """Send a forward and save its specs for backward."""
        if not is_last_stage:
            self._exchange(output=output_tensors, send_forward_chunk_id=send_chunk_id)

    def send_forward_recv_backward(
        self,
        output_tensors: PipelinePayload,
        tensor_shapes: object,
        is_last_stage: bool,
        *,
        send_chunk_id: int = 0,
        recv_chunk_id: int = 0,
    ) -> PipelineGradientMessage | None:
        """Send this forward while receiving the oldest pending forward's gradients."""
        if not is_last_stage:
            return self._exchange(
                output=output_tensors,
                recv_backward=True,
                send_forward_chunk_id=send_chunk_id,
                recv_backward_chunk_id=recv_chunk_id,
            )[1]

    def recv_backward(
        self, tensor_shapes: object, is_last_stage: bool, *, recv_chunk_id: int = 0
    ) -> PipelineGradientMessage | None:
        """Receive the oldest pending forward's gradients during cooldown."""
        if not is_last_stage:
            return self._exchange(recv_backward=True, recv_backward_chunk_id=recv_chunk_id)[1]

    def send_backward(
        self, input_tensor_grads: PipelineGradients, is_first_stage: bool, *, send_chunk_id: int = 0
    ) -> None:
        """Return gradients in the incoming forward's field order."""
        if not is_first_stage:
            self._exchange(gradients=input_tensor_grads, send_backward_chunk_id=send_chunk_id)

    def send_backward_recv_forward(
        self,
        input_tensor_grads: PipelineGradients,
        tensor_shapes: object,
        is_first_stage: bool,
        *,
        send_chunk_id: int = 0,
        recv_chunk_id: int = 0,
    ) -> PipelinePayload | None:
        """Return old gradients while receiving a new forward."""
        if not is_first_stage:
            return self._exchange(
                gradients=input_tensor_grads,
                recv_forward=True,
                send_backward_chunk_id=send_chunk_id,
                recv_forward_chunk_id=recv_chunk_id,
            )[0]

    def send_forward_recv_forward(
        self,
        output_tensor: PipelinePayload | None,
        recv_prev: bool,
        tensor_shape: object,
        overlap_p2p_comm: bool = False,
        *,
        send_chunk_id: int = 0,
        recv_chunk_id: int = 0,
    ) -> PipelinePayload | None:
        """Exchange a VPP chunk's forward payload with its physical neighbor."""
        if overlap_p2p_comm:
            raise ValueError("Typed VPP boundaries do not support overlapping p2p communication")
        return self._exchange(
            output=output_tensor,
            recv_forward=recv_prev,
            send_forward_chunk_id=send_chunk_id,
            recv_forward_chunk_id=recv_chunk_id,
        )[0]

    def send_backward_recv_backward(
        self,
        input_tensor_grad: PipelineGradients | PipelineGradientMessage | None,
        recv_next: bool,
        tensor_shape: object,
        overlap_p2p_comm: bool = False,
        *,
        send_chunk_id: int = 0,
        recv_chunk_id: int = 0,
    ) -> PipelineGradientMessage | None:
        """Exchange a VPP chunk's backward gradients with its physical neighbor."""
        if overlap_p2p_comm:
            raise ValueError("Typed VPP boundaries do not support overlapping p2p communication")
        return self._exchange(
            gradients=input_tensor_grad,
            recv_backward=recv_next,
            send_backward_chunk_id=send_chunk_id,
            recv_backward_chunk_id=recv_chunk_id,
        )[1]

    def send_forward_backward_recv_forward_backward(
        self,
        output_tensor: PipelinePayload | None,
        input_tensor_grad: PipelineGradients | PipelineGradientMessage | None,
        recv_prev: bool,
        recv_next: bool,
        tensor_shape: object,
        *,
        send_forward_chunk_id: int = 0,
        send_backward_chunk_id: int = 0,
        recv_forward_chunk_id: int = 0,
        recv_backward_chunk_id: int = 0,
    ) -> tuple[PipelinePayload | None, PipelineGradientMessage | None]:
        """Exchange forward and backward payloads for different VPP chunks."""
        return self._exchange(
            output=output_tensor,
            gradients=input_tensor_grad,
            recv_forward=recv_prev,
            recv_backward=recv_next,
            send_forward_chunk_id=send_forward_chunk_id,
            send_backward_chunk_id=send_backward_chunk_id,
            recv_forward_chunk_id=recv_forward_chunk_id,
            recv_backward_chunk_id=recv_backward_chunk_id,
        )

    def finish(self) -> None:
        """Drain outstanding P2P and check that every microbatch completed backward."""
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
