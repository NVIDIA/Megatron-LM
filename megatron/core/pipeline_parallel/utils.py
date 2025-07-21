# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Callable, Optional

import torch
from torch.autograd import Variable

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed.custom_fsdp import FullyShardedDataParallel as custom_FSDP
from megatron.core.extensions.transformer_engine import TE_MODULE_CLASSNAMES
from megatron.core.transformer.module import Float16Module
from megatron.core.utils import get_pg_rank, get_pg_size, make_viewless_tensor

try:
    from megatron.core.distributed import TorchFullyShardedDataParallel as torch_FSDP

    ALL_MODULE_WRAPPER_CLASSNAMES = (DDP, torch_FSDP, custom_FSDP, Float16Module)
except ImportError:
    ALL_MODULE_WRAPPER_CLASSNAMES = (DDP, custom_FSDP, Float16Module)


def is_pp_first_stage(pp_group: torch.distributed.ProcessGroup):
    """Return True if in the first pipeline model-parallel stage, False otherwise."""
    return get_pg_rank(pp_group) == 0


def is_pp_last_stage(pp_group: torch.distributed.ProcessGroup):
    """Return True if in the last pipeline-model-parallel stage, False otherwise."""
    return get_pg_rank(pp_group) == (get_pg_size(pp_group) - 1)


def is_vp_first_stage(vp_stage: int, vp_size: int | None):
    """Return True if in the first virtual pipeline model-parallel stage, False otherwise."""
    if vp_size is None or vp_size <= 1:
        return True
    return vp_stage == 0


def is_vp_last_stage(vp_stage: int, vp_size: int | None):
    """Return True if in the last virtual pipeline model-parallel stage, False otherwise."""
    if vp_size is None or vp_size <= 1:
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
        for g in output_grad:
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

    @classmethod
    @abstractmethod
    def run(
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
    if _COMP_STREAM is not None and _COM_STREAM is not None:
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


def unwrap_model(model, module_instances=ALL_MODULE_WRAPPER_CLASSNAMES):
    """Unwrap_model to return the final model instance"""
    from megatron.core.models.gpt.gpt_model import GPTModel

    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while isinstance(model_module, module_instances):
            model_module = model_module.module
        assert isinstance(
            model_module, GPTModel
        ), "The final unwrapped model must be a GPTModel instance"
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model


def register_wgrad_accumulation_and_reduce_func(model):
    """Register the wgrad accumulation and reduce function for the TE modules"""
    if not isinstance(model, list):
        model = [model]
    for i in range(len(model)):
        assert isinstance(model[i], DDP), "Only DDP wrapper is supported now."
    unwrapped_model = unwrap_model(model)
    for i in range(len(unwrapped_model)):
        for name, module in unwrapped_model[i].named_modules():
            if isinstance(module, TE_MODULE_CLASSNAMES):
                if hasattr(module, 'register_wgrad_accumulation_and_reduce_func'):
                    module.register_wgrad_accumulation_and_reduce_func(
                        model[i]._make_backward_post_hook
                    )
                else:
                    warnings.warn(
                        f"register_wgrad_accumulation_and_reduce_func is not found for {name}, "
                        "skip the registration"
                    )
