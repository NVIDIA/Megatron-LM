# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from contextlib import contextmanager

import torch
from torch.autograd import Variable

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.module import Float16Module
from megatron.core.utils import make_viewless_tensor


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


class FakeScheduleNode:
    """A placeholder node in the computation graph that simply passes through inputs and outputs.

    This class is used as a no-op node in the scheduling system when a real computation node
    is not needed but the interface must be maintained. It simply returns its inputs unchanged
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
        forward_func,
        stream,
        event,
        backward_func=None,
        free_input=False,
        name="schedule_node",
    ):
        """Initialize a schedule node.

        Args:
            forward_func (callable): Function to execute during forward pass
            stream (torch.cuda.Stream): CUDA stream for computation
            event (torch.cuda.Event): Event for synchronization
            backward_func (callable, optional): Function for backward pass
            memory_strategy (MemoryManagementStrategy, optional): Strategy for memory management
            name (str): Name of the node for debugging
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
        for g in output_grad:
            g.record_stream(self.stream)

        return self.get_grad()

    def get_grad(self):
        """Get the grad of inputs"""
        grad = tuple([e.grad if e is not None else None for e in self.inputs])
        # clear state
        self.inputs = None
        self.output = None
        # multiple in, multiple out
        if len(grad) == 1:
            grad = grad[0]
        return grad


class AbstractSchedulePlan(ABC):
    """To use combined 1f1b, model must implement build_schedule_plan while take the same
    signature as model forward but return an instance of AbstractSchedulePlan"""

    @classmethod
    @abstractmethod
    def forward_backward(
        cls,
        f_schedule_plan,
        b_schedule_plan,
        grad=None,
        pre_forward=None,
        pre_backward=None,
        post_forward=None,
        post_backward=None,
    ):
        """forward_backward is the protocol between our schedule logic and model"""
        ...


_COMP_STREAM = None
_COM_STREAM = None


def set_streams(comp_stream=None, com_stream=None):
    """Set the streams for communication and computation"""
    global _COMP_STREAM
    global _COM_STREAM
    if _COMP_STREAM is not None:
        return

    if comp_stream is None:
        comp_stream = torch.cuda.current_stream()
    if com_stream is None:
        com_stream = torch.cuda.Stream(device="cuda")

    assert _COMP_STREAM is None
    assert _COM_STREAM is None
    _COMP_STREAM = comp_stream
    _COM_STREAM = com_stream


def get_comp_stream():
    """Get the stream for computation"""
    global _COMP_STREAM
    return _COMP_STREAM


def get_com_stream():
    """Get the stream for communication"""
    global _COM_STREAM
    return _COM_STREAM


class VppContextManager:
    """A reusable context manager for switch vpp stage"""

    def __init__(self, vpp_rank):
        self.vpp_rank = vpp_rank

    def __enter__(self):
        self.origin_vpp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
        parallel_state.set_virtual_pipeline_model_parallel_rank(self.vpp_rank)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        parallel_state.set_virtual_pipeline_model_parallel_rank(self.origin_vpp_rank)



def unwrap_model(model):
    """Unwrap_model DistributedDataParallel and Float16Module wrapped model
    to return GPTModel instance
    """
    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while isinstance(model_module, (DistributedDataParallel, Float16Module)):
            model_module = model_module.module
        assert isinstance(
            model_module, GPTModel
        ), "The final unwrapped model must be a GPTModel instance"
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model


def wrap_forward_func(forward_step_func):
    """Wrap the input to forward_step_func.
    The wrapped function will return forward_schedule_plan and the loss_function.
    """

    def wrapped_func(data_iterator, model):
        # Model is unwrapped to get GPTModel instance.
        # GPTModel.build_schedule_plan(model_forward_inputs) is called in the forward_step.
        # The return value becomes (forward_schedule_plan, loss_function),
        # which is used to be (forward_output_tensor, loss_function).
        return forward_step_func(data_iterator, unwrap_model(model).build_schedule_plan)

    return wrapped_func
