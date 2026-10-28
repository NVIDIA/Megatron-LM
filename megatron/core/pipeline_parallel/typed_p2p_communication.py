# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Typed P2P for ordinary and interleaved 1F1B.

Each schedule invocation owns per-chunk FIFOs of host descriptors, never activation
graphs. Every message is described by a prepared ``PipelinePayloadSpec``, so all
receives are posted without runtime shape discovery and peers exchange only tensor
data: no headers or control messages, and no host synchronization.
"""

from collections import deque

import torch
import torch.distributed as dist

from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.pipeline_payload import (
    PipelineGradients,
    PipelinePayloadPlan,
    PipelinePayloadSpec,
    TensorStatePayload,
)


class _P2PWork:
    """Keep one direction's wire buffers alive until all its requests are waited."""

    def __init__(self, buffers):
        self.requests, self.buffers = [], buffers

    def wait(self) -> None:
        """Wait every request once and release the detached communication buffers."""
        for request in self.requests:
            request.wait()
        self.requests.clear()
        self.buffers.clear()


class TypedP2PCommunicator(P2PCommunicator):
    """Exchange ordered tensor fields described by per-microbatch payload plans."""

    def __init__(
        self,
        pp_group: dist.ProcessGroup,
        config: ModelParallelConfig,
        make_payload,
        payload_plans: list[PipelinePayloadPlan],
        *,
        forward_only: bool = False,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(pp_group, config)
        if config.overlap_p2p_comm_warmup_flush:
            raise ValueError("Typed P2P warmup/flush prefetch is not supported")
        if config.use_ring_exchange_p2p:
            raise ValueError("Typed P2P does not support ring_exchange")
        self.make_payloads = make_payload if isinstance(make_payload, list) else [make_payload]
        num_chunks = config.virtual_pipeline_model_parallel_size or 1
        if config.overlap_p2p_comm and (num_chunks < 2 or config.batch_p2p_comm):
            raise ValueError("Typed P2P overlap requires VPP and batch_p2p_comm=False")
        if len(self.make_payloads) != num_chunks or not all(map(callable, self.make_payloads)):
            raise ValueError("Typed P2P requires a payload factory for every virtual chunk")
        if len(payload_plans) != num_chunks or not all(
            isinstance(plan, PipelinePayloadPlan) for plan in payload_plans
        ):
            raise ValueError("Typed P2P requires a payload plan for every virtual chunk")
        self.payload_plans = payload_plans
        self.forward_only = forward_only
        self.device = (
            torch.device("cuda", torch.cuda.current_device()) if device is None else device
        )
        # As in the tensor-only communicator, PP=2 carries one physical direction on
        # WORLD so a posted receive never queues ahead of a send to the same peer.
        self._reverse_group = (
            dist.group.WORLD
            if pp_group.size() == 2 and dist.get_backend(pp_group) != "ucc"
            else pp_group
        )
        self._sent = [deque() for _ in range(num_chunks)]
        self._received = [deque() for _ in range(num_chunks)]
        self._send_microbatch = [0] * num_chunks
        self._recv_microbatch = [0] * num_chunks
        self._pending: list[_P2PWork] = []
        self._validate_plan_peers()

    def _validate_plan_peers(self):
        """Check once per invocation that each boundary's sender and receiver plans agree."""
        local = tuple(
            tuple(
                tuple(None if spec is None else spec.fingerprint for spec in descriptors)
                for descriptors in (plan.incoming, plan.outgoing)
            )
            for plan in self.payload_plans
        )
        plans = [None] * self.pp_group.size()
        dist.all_gather_object(plans, local, group=self.pp_group)
        if any(len(plan) != len(local) for plan in plans):
            raise ValueError("Pipeline peers disagree on their chunk count")
        ordered = [plans[rank][chunk] for chunk in range(len(local)) for rank in range(len(plans))]
        for left, right in zip(ordered, ordered[1:]):
            if left[1] != right[0]:
                raise ValueError("Pipeline peer schemas do not match")

    def _planned(self, chunk_id, microbatch, *, outgoing) -> PipelinePayloadSpec:
        plan = self.payload_plans[chunk_id]
        descriptors = plan.outgoing if outgoing else plan.incoming
        if not 0 <= microbatch < len(descriptors) or descriptors[microbatch] is None:
            raise ValueError(
                f"Missing pipeline payload plan for chunk {chunk_id}, microbatch {microbatch}"
            )
        return descriptors[microbatch]

    def _transfer(self, *, send_next=None, send_prev=None, recv_prev=None, recv_next=None):
        directions = [
            ("send_next", send_next, self.next_rank, dist.isend),
            ("send_prev", send_prev, self.prev_rank, dist.isend),
            ("recv_prev", recv_prev, self.prev_rank, dist.irecv),
            ("recv_next", recv_next, self.next_rank, dist.irecv),
        ]
        # Forward precedes backward in each peer's message order, and odd stages receive
        # first, matching the tensor-only communicator's deadlock-free pairing.
        if self.current_stage % 2:
            directions = directions[2:] + directions[:2]
        handles, batch = {}, []
        for name, tensors, peer, operation in directions:
            if tensors is None:
                continue
            buffers = [
                tensor.detach().contiguous() if operation is dist.isend else tensor
                for tensor in tensors
                if tensor is not None and tensor.numel()
            ]
            handle = handles[name] = _P2PWork(buffers)
            self._pending.append(handle)
            even_sender = (self.current_stage % 2 == 0) == (operation is dist.isend)
            group = self.pp_group if even_sender else self._reverse_group
            for tensor in buffers:
                if self.config.batch_p2p_comm:
                    batch.append((dist.P2POp(operation, tensor, peer, self.pp_group), handle))
                else:
                    handle.requests.append(operation(tensor, peer, group=group))
        if batch:
            requests = dist.batch_isend_irecv([op for op, _ in batch])
            if len(requests) == len(batch):
                for request, (_, handle) in zip(requests, batch):
                    handle.requests.append(request)
            else:
                # NCCL can return one coalesced Work for the whole batch; every direction
                # must wait it before releasing its buffers.
                for handle in handles.values():
                    handle.requests.extend(requests)
        return handles

    def _gradient_wire(self, gradients: PipelineGradients, spec: PipelinePayloadSpec):
        fields = spec.schema.packed_fields
        if len(gradients) != len(fields):
            raise ValueError("Pipeline gradients must match the incoming field slots")
        wire = []
        for field, grad in zip(fields, gradients):
            if not field.differentiable:
                if grad is not None:
                    raise ValueError(f"Unexpected gradient for pipeline field {field.key}")
                continue
            if grad is None:
                if 0 in field.shape:
                    continue
                # Zero-filling would run producer-side auxiliary-loss hooks for an unused
                # branch, so an unused field must be declared non-differentiable instead.
                raise ValueError(
                    f"Pipeline field {field.key} is differentiable but received no gradient; "
                    "declare it non-differentiable in this microbatch's plan"
                )
            if tuple(grad.shape) != field.shape or grad.dtype != field.dtype:
                raise ValueError(f"Invalid gradient for pipeline field {field.key}")
            if grad.device != self.device:
                raise ValueError(f"Invalid gradient device for pipeline field {field.key}")
            wire.append(grad)
        return wire

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
        timers = self.config.timers
        timer = None if timers is None else timers("-".join(operations), log_level=2)
        if timer is not None:
            timer.start()
        try:
            forward_sends = backward_sends = received = grads = None
            if output is not None:
                microbatch = self._send_microbatch[forward_send_chunk_id]
                spec = self._planned(forward_send_chunk_id, microbatch, outgoing=True)
                if not isinstance(output, TensorStatePayload) or output.spec != spec:
                    raise ValueError(
                        f"Pipeline chunk {forward_send_chunk_id}, microbatch {microbatch}: "
                        f"outgoing {getattr(output, 'spec', output)} disagrees with plan {spec}"
                    )
                spec.schema.validate(output.tensors)
                if output.device != self.device:
                    raise ValueError("Expected a pipeline payload on the communication device")
                forward_sends = output.tensors
                self._send_microbatch[forward_send_chunk_id] += 1
                if not self.forward_only:
                    self._sent[forward_send_chunk_id].append(spec)
            if gradients is not None:
                pending = self._received[backward_send_chunk_id]
                backward_sends = self._gradient_wire(gradients, pending[0])
                pending.popleft()
            if recv_backward:
                spec = self._sent[backward_recv_chunk_id].popleft()
                grads = tuple(
                    (
                        torch.empty(field.shape, dtype=field.dtype, device=self.device)
                        if field.differentiable
                        else None
                    )
                    for field in spec.schema.packed_fields
                )
            if recv_forward:
                microbatch = self._recv_microbatch[forward_recv_chunk_id]
                self._recv_microbatch[forward_recv_chunk_id] += 1
                spec = self._planned(forward_recv_chunk_id, microbatch, outgoing=False)
                tensors = tuple(
                    torch.empty(
                        field.shape,
                        dtype=field.dtype,
                        device=self.device,
                        requires_grad=field.differentiable,
                    )
                    for field in spec.schema.packed_fields
                )
                received = self.make_payloads[forward_recv_chunk_id](tensors, spec)
                if not self.forward_only:
                    self._received[forward_recv_chunk_id].append(spec)
            handles = self._transfer(
                send_next=forward_sends,
                send_prev=backward_sends,
                recv_prev=None if received is None else received.tensors,
                recv_next=grads,
            )
            # The detached wire buffers own the sent storage until their Work completes.
            if output is not None and self.config.deallocate_pipeline_outputs:
                output.release_output()
            if overlap:
                return received, grads, handles
            for handle in handles.values():
                handle.wait()
            if (
                self.config.batch_p2p_comm
                and self.config.batch_p2p_sync
                and self.device.type == "cuda"
            ):
                torch.cuda.synchronize()
            return received, grads
        finally:
            if timer is not None:
                timer.stop()

    def recv_forward(self, tensor_shapes: object, is_first_stage: bool):
        """Receive the next forward payload."""
        if not is_first_stage:
            return self._exchange(recv_forward=True)[0]

    def send_forward(self, output_tensors: TensorStatePayload, is_last_stage: bool) -> None:
        """Send a forward and save its descriptor for backward."""
        if not is_last_stage:
            self._exchange(output=output_tensors)

    def send_forward_recv_backward(
        self, output_tensors: TensorStatePayload, tensor_shapes: object, is_last_stage: bool
    ) -> PipelineGradients | None:
        """Send this forward while receiving the oldest pending forward's gradients."""
        if not is_last_stage:
            return self._exchange(output=output_tensors, recv_backward=True)[1]

    def recv_backward(
        self, tensor_shapes: object, is_last_stage: bool, *, recv_chunk_id: int = 0
    ) -> PipelineGradients | None:
        """Receive the oldest pending forward's gradients during cooldown."""
        if not is_last_stage:
            return self._exchange(recv_backward=True, backward_recv_chunk_id=recv_chunk_id)[1]

    def send_backward(self, input_tensor_grads: PipelineGradients, is_first_stage: bool) -> None:
        """Return gradients in the incoming forward's field order."""
        if not is_first_stage:
            self._exchange(gradients=input_tensor_grads)

    def send_backward_recv_forward(
        self, input_tensor_grads: PipelineGradients, tensor_shapes: object, is_first_stage: bool
    ) -> TensorStatePayload | None:
        """Return old gradients while receiving a new forward."""
        if not is_first_stage:
            return self._exchange(gradients=input_tensor_grads, recv_forward=True)[0]

    def send_forward_recv_forward(
        self,
        output_tensor,
        recv_prev: bool,
        tensor_shape: object,
        overlap_p2p_comm: bool = False,
        *,
        send_chunk_id: int = 0,
        recv_chunk_id: int = 0,
    ):
        """Exchange forwards; with overlap, wait the returned handle before model use."""
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
        input_tensor_grad,
        recv_next: bool,
        tensor_shape: object,
        overlap_p2p_comm: bool = False,
        *,
        send_chunk_id: int = 0,
        recv_chunk_id: int = 0,
    ):
        """Exchange gradients; with overlap, wait the returned handle before backward."""
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
        output_tensor,
        input_tensor_grad,
        recv_prev: bool,
        recv_next: bool,
        tensor_shape: object,
        *,
        forward_send_chunk_id: int = 0,
        backward_send_chunk_id: int = 0,
        forward_recv_chunk_id: int = 0,
        backward_recv_chunk_id: int = 0,
    ):
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
        """Drain outstanding P2P and check that every planned microbatch completed."""
        for handle in self._pending:
            handle.wait()
        self._pending.clear()
        if any(self._sent) or any(self._received):
            raise RuntimeError("Typed pipeline schedule left unfinished microbatches")
        for chunk, plan in enumerate(self.payload_plans):
            if self._send_microbatch[chunk] != sum(s is not None for s in plan.outgoing) or (
                self._recv_microbatch[chunk] != sum(s is not None for s in plan.incoming)
            ):
                raise RuntimeError("Typed pipeline schedule did not consume its payload plan")
