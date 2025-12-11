# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Callable, Optional

import torch
from torch.autograd import Variable

from megatron.core.utils import get_pg_rank, get_pg_size, make_viewless_tensor


def is_pp_first_stage(pp_group: torch.distributed.ProcessGroup):
    """Return True if in the first pipeline model-parallel stage, False otherwise."""
    return get_pg_rank(pp_group) == 0


def is_pp_last_stage(pp_group: torch.distributed.ProcessGroup):
    """Return True if in the last pipeline-model-parallel stage, False otherwise."""
    return get_pg_rank(pp_group) == (get_pg_size(pp_group) - 1)


def is_vp_first_stage(vp_stage: int, vp_size: int | None):
    """Return True if in the first virtual pipeline model-parallel stage, False otherwise."""
    if vp_size is None or vp_size <= 1:
        assert vp_stage is None or vp_stage == 0, (
            f"Expected vp_stage to be 0 or None when vp_size is <= 1 or None, "
            f"but got vp_stage={vp_stage} and vp_size={vp_size}"
        )
        return True
    return vp_stage == 0


def is_vp_last_stage(vp_stage: int, vp_size: int | None):
    """Return True if in the last virtual pipeline model-parallel stage, False otherwise."""
    if vp_size is None or vp_size <= 1:
        assert vp_stage is None or vp_stage == 0, (
            f"Expected vp_stage to be 0 or None when vp_size is <= 1 or None, "
            f"but got vp_stage={vp_stage} and vp_size={vp_size}"
        )
        return True
    return vp_stage == (vp_size - 1)


def get_pp_first_rank(pp_group: torch.distributed.ProcessGroup):
    """Return the global rank of the first rank in the pipeline parallel group."""
    pp_ranks = torch.distributed.get_process_group_ranks(pp_group)
    return pp_ranks[0]


def get_pp_last_rank(pp_group: torch.distributed.ProcessGroup):
    """Return the global rank of the last rank in the pipeline parallel group."""
    pp_ranks = torch.distributed.get_process_group_ranks(pp_group)
    return pp_ranks[-1]


def get_pp_next_rank(pp_group: torch.distributed.ProcessGroup):
    """Return the global rank of the next rank in the pipeline parallel group, or None if last
    stage."""
    if is_pp_last_stage(pp_group):
        return None
    current_rank_in_group = get_pg_rank(pp_group)
    pp_ranks = torch.distributed.get_process_group_ranks(pp_group)
    return pp_ranks[current_rank_in_group + 1]


def get_pp_prev_rank(pp_group: torch.distributed.ProcessGroup):
    """Return the global rank of the previous rank in the pipeline parallel group, or None if
    first stage."""
    if is_pp_first_stage(pp_group):
        return None
    current_rank_in_group = get_pg_rank(pp_group)
    pp_ranks = torch.distributed.get_process_group_ranks(pp_group)
    return pp_ranks[current_rank_in_group - 1]


def make_viewless(e):
    """Make_viewless util func"""
    e = make_viewless_tensor(inp=e, requires_grad=e.requires_grad, keep_graph=True)
    return e


@contextmanager
def stream_acquire_context(stream, event):
    """Stream acquire context"""
    event.wait(stream)
    try:
        yield
    finally:
        event.record(stream)


class NoopScheduleNode:
    """A placeholder node in the computation graph that simply passes through inputs and outputs.

    This class is used as a no-op node in the scheduling system when a real computation node
    is not needed but the interface must be maintained (e.g., dense layer doesn't need
    moe_dispatch and moe_combine). It simply returns its inputs unchanged
    in both forward and backward passes.
    """

    def forward(self, inputs):
        """Passes through inputs unchanged in the forward pass."""
        return inputs

    def backward(self, outgrads):
        """Passes through gradients unchanged in the backward pass."""
        return outgrads


class ScheduleNode:
    """Base node for fine-grained scheduling.

    This class represents a computational node in the pipeline schedule.
    It handles the execution of forward and backward operations on a stream.
    """

    def __init__(
        self,
        forward_func: Callable,
        stream: torch.cuda.Stream,
        event: torch.cuda.Event,
        backward_func: Optional[Callable] = None,
        free_input: bool = False,
        name: str = "schedule_node",
    ):
        """Initialize a schedule node.

        Args:
            forward_func (callable): Function to execute during the forward pass.
            stream (torch.cuda.Stream): The CUDA stream for this node's computation.
                This can be either a 'compute' stream or a 'communicate' stream.
                - 'compute' stream: Used for computational nodes like attention and experts.
                - 'communicate' stream: Used for nodes that handle token communication,
                  such as token dispatch and combine operations in MoE layers.
            event (torch.cuda.Event): The CUDA event used for synchronization. Each
                microbatch within a model chunk shares the same event, which is used
                to manage dependencies between nodes operating on different streams.
            backward_func (callable, optional): Function for the backward pass.
            free_input (bool): Flag to indicate if the input should be freed after the
                forward pass.
            name (str): Name of the node for debugging purposes.
        """
        self.name = name
        self.forward_func = forward_func
        self.backward_func = backward_func if backward_func else self.default_backward_func
        self.stream = stream
        self.event = event
        self.free_input = free_input
        self.inputs = None
        self.outputs = None

    def default_backward_func(self, outputs, output_grad):
        """Default backward function"""
        Variable._execution_engine.run_backward(
            tensors=outputs,
            grad_tensors=output_grad,
            keep_graph=False,
            create_graph=False,
            inputs=tuple(),
            allow_unreachable=True,
            accumulate_grad=True,
        )
        return output_grad

    def forward(self, inputs=()):
        """Schedule node forward"""
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        return self._forward(*inputs)

    def _forward(self, *inputs):
        with stream_acquire_context(self.stream, self.event):
            torch.cuda.nvtx.range_push(f"{self.name} forward")
            with torch.cuda.stream(self.stream):
                self.inputs = [make_viewless(e).detach() if e is not None else None for e in inputs]
                for i, input in enumerate(self.inputs):
                    if input is not None:
                        input.requires_grad = inputs[i].requires_grad

                data = tuple(self.inputs)
                data = self.forward_func(*data)

                if not isinstance(data, tuple):
                    data = make_viewless(data)
                else:
                    data = tuple(
                        [make_viewless(e) if isinstance(e, torch.Tensor) else e for e in data]
                    )

                self.output = data
            torch.cuda.nvtx.range_pop()

        # Immediately frees input tensors after they are used for nodes
        # where inputs are no longer needed after computation.
        if self.free_input:
            for input in inputs:
                if input is not None:
                    input.record_stream(self.stream)
                    input.untyped_storage().resize_(0)

        return self.output

    def get_output(self):
        """Get the forward output"""
        return self.output

    def backward(self, output_grad):
        """Schedule node backward"""
        if not isinstance(output_grad, tuple):
            output_grad = (output_grad,)
        return self._backward(*output_grad)

    def _backward(self, *output_grad):
        with stream_acquire_context(self.stream, self.event):
            torch.cuda.nvtx.range_push(f"{self.name} backward")
            with torch.cuda.stream(self.stream):
                outputs = self.output
                if not isinstance(outputs, tuple):
                    outputs = (outputs,)
                assert len(outputs) == len(output_grad), (
                    f"{len(outputs)} of {type(outputs[0])} is not equal to "
                    f"{len(output_grad)} of {type(output_grad[0])}"
                )
                output_grad = self.backward_func(outputs, output_grad)
            torch.cuda.nvtx.range_pop()

        # output_grad maybe from another stream
        if output_grad:
            for g in output_grad:
                if g is not None:
                    g.record_stream(self.stream)

        grads = self.get_grad()
        self._release_state()

        return grads

    def get_grad(self):
        """Get the grad of inputs"""
        grad = tuple([e.grad if e is not None else None for e in self.inputs])
        # multiple in, multiple out
        if len(grad) == 1:
            grad = grad[0]
        return grad

    def _release_state(self):
        """Clear the state of the node"""
        self.inputs = None
        self.output = None
        del self.forward_func
        del self.backward_func


class AbstractSchedulePlan(ABC):
    """To use combined 1f1b, model must implement build_schedule_plan while take the same
    signature as model forward but return an instance of AbstractSchedulePlan"""

    @staticmethod
    @abstractmethod
    def run(
        f_schedule_plan,
        b_schedule_plan,
        grad=None,
        pre_forward=None,
        pre_backward=None,
        post_forward=None,
        post_backward=None,
    ):
        """run() is the protocol between our schedule logic and model, which is used to schedule
        the forward and backward schedule plans for the models.
        """
        ...


_COMP_STREAM = None
_COMM_STREAM = None


def set_streams(comp_stream=None, comm_stream=None):
    """Set the streams for communication and computation"""
    global _COMP_STREAM
    global _COMM_STREAM
    if _COMP_STREAM is not None and _COMM_STREAM is not None:
        return

    if comp_stream is None:
        comp_stream = torch.cuda.current_stream()
    if comm_stream is None:
        comm_stream = torch.cuda.Stream(device="cuda")

    assert _COMP_STREAM is None
    assert _COMM_STREAM is None
    _COMP_STREAM = comp_stream
    _COMM_STREAM = comm_stream


def get_comp_stream():
    """Get the stream for computation"""
    global _COMP_STREAM
    return _COMP_STREAM


def get_comm_stream():
    """Get the stream for communication"""
    global _COMM_STREAM
    return _COMM_STREAM
