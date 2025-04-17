# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import contextlib
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import List, Union

import torch
from torch import Tensor
from torch.autograd.variable import Variable

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.moe.router import MoEAuxLossAutoScaler
from megatron.core.utils import get_attr_wrapped_model, make_viewless_tensor

# Types
Shape = Union[List[int], torch.Size]


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
        memory_strategy=None,
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
        self.memory_strategy = memory_strategy or NoOpMemoryStrategy()
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
                    data = tuple([make_viewless(e) if isinstance(e, Tensor) else e for e in data])

                self.output = data
            torch.cuda.nvtx.range_pop()

        # Handle inputs using the memory strategy
        self.memory_strategy.handle_inputs(inputs, self.stream)

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
        f_context=None,
        b_context=None,
        pre_forward=None,
        pre_backward=None,
        post_forward=None,
        post_backward=None,
    ):
        """forward_backward is the protocol between our schedule logic and model"""
        ...


def schedule_chunk_1f1b(
    f_schedule_plan,
    b_schedule_plan,
    grad=None,
    f_context=None,
    b_context=None,
    pre_forward=None,
    pre_backward=None,
    post_forward=None,
    post_backward=None,
):
    """Model level 1f1b fine-grained schedule

    This function schedules the forward and backward passes for a chunk of the model.
    It takes in the forward schedule plan, backward schedule plan, gradient, and optional
    context managers for the forward and backward passes.

    Args:
        f_schedule_plan (subclass of AbstractSchedulePlan): The forward schedule plan
        b_schedule_plan (subclass of AbstractSchedulePlan): The backward schedule plan
        grad (Tensor or None): The gradient of the loss function
        f_context (VppContextManager or None): The VppContextManager for the forward pass
        b_context (VppContextManager or None): The VppContextManager for the backward pass
        pre_forward (callable or None): The function to call before the forward pass
        pre_backward (callable or None): The function to call before the backward pass
        post_forward (callable or None): The function to call after the forward pass
        post_backward (callable or None): The function to call after the backward pass

    Returns:
        The output of the forward pass
    """

    # Calls fine_grained_schedule.py::ModelChunkSchedulePlan.forward_backward(),
    # which calls fine_grained_schedule.py::schedule_chunk_1f1b()
    return type(f_schedule_plan or b_schedule_plan).forward_backward(
        f_schedule_plan,
        b_schedule_plan,
        grad=grad,
        f_context=f_context,
        b_context=b_context,
        pre_forward=pre_forward,
        pre_backward=pre_backward,
        post_forward=post_forward,
        post_backward=post_backward,
    )


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


def forward_backward_step(
    forward_step_func,
    data_iterator,
    f_model,
    num_microbatches,
    input_tensor,
    forward_data_store,
    b_model,
    b_input_tensor,
    b_output_tensor,
    b_output_tensor_grad,
    config,
    f_context=None,
    b_context=None,
    pre_forward=None,
    pre_backward=None,
    post_forward=None,
    post_backward=None,
    collect_non_loss_data=False,
    checkpoint_activations_microbatch=None,
    is_first_microbatch=False,
    current_microbatch=None,
    encoder_decoder_xattn=False,
):
    """Merged forward and backward step for combined_1f1b.

    Args:
        Need to accept the argument of both forward_step() and backward_step().
        forward_step_func (callable): is wrapped by wrap_forward_func() which is now returning
            a forward schedule plan which is an input of schedule_chunk_1f1b function.
        f_context (VppContextManager or nullcontext): The context manager for setting vpp ranks.
        b_context (VppContextManager or nullcontext): The context manager for setting vpp ranks.

        Only exists in 1f1b steady state with p2p overlap.
            pre_forward (callable): The function to call before the forward_step.
            pre_backward (callable): The function to call before the backward_step.
            post_forward (callable): The function to call after the forward_step.
            post_backward (callable): The function to call after the backward_step.

    Returns:
        forward_output_tensor (Tensor or list[Tensor]): The output object(s) from the forward step.
        forward_num_tokens (Tensor): The number of tokens.
        backward_input_tensor_grad (Tensor): The grad of the input tensor.

    Descriptions:
        This method merges the forward_step() and backward_step() methods in the schedules.py file.
        Assuming that:
            def forward_step():
                # forward_preprocess()
                # forward_compute()
                # forward_postprocess()
            def backward_step():
                # backward_preprocess()
                # backward_compute()
                # backward_postprocess()
        Then the forward_backward_step() method will be:
            def forward_backward_step():
                # forward_preprocess() // the same as the forward_step()
                # GENERATE f_schedule_plan // schedule happens in schedule_chunk_1f1b()
                # backward_preprocess() // the same as the backward_step()
                # COMBINED_FORWARD_BACKWARD_COMPUTE() // by calling schedule_chunk_1f1b()
                # forward_postprocess() // the same as the forward_step()
                # backward_postprocess() // the same as the backward_step()
    """
    assert (
        checkpoint_activations_microbatch is None
    ), "checkpoint_activations_microbatch is not supported for combined_1f1b"

    if config.combined_1f1b_recipe != "ep_a2a":
        raise NotImplementedError(
            f"combined_1f1b_recipe {config.combined_1f1b_recipe} not supported yet"
        )

    from .schedules import set_current_microbatch

    if f_model is not None and config.timers is not None:
        config.timers('forward-compute', log_level=2).start()

    if config.enable_autocast:
        context_manager = torch.autocast("cuda", dtype=config.autocast_dtype)
    else:
        context_manager = contextlib.nullcontext()

    # forward preprocess
    unwrap_output_tensor = False
    f_schedule_plan = None
    if f_model is not None:
        with f_context:
            if is_first_microbatch and hasattr(f_model, 'set_is_first_microbatch'):
                f_model.set_is_first_microbatch()
            if current_microbatch is not None:
                set_current_microbatch(f_model, current_microbatch)
            if not isinstance(input_tensor, list):
                input_tensor = [input_tensor]
                unwrap_output_tensor = True

            set_input_tensor = get_attr_wrapped_model(f_model, "set_input_tensor")
            set_input_tensor(input_tensor)

            with context_manager:  # autocast context
                f_schedule_plan, loss_func = forward_step_func(data_iterator, f_model)
                assert isinstance(
                    f_schedule_plan, AbstractSchedulePlan
                ), "first output of forward_step_func must be one instance of AbstractSchedulePlan"

    # backward preprocess
    unwrap_input_tensor_grad = False
    b_schedule_plan = None
    if b_model is not None:
        # Retain the grad on the input_tensor.
        if not isinstance(b_input_tensor, list):
            b_input_tensor = [b_input_tensor]
            unwrap_input_tensor_grad = True
        for x in b_input_tensor:
            if x is not None:
                x.retain_grad()

        if not isinstance(b_output_tensor, list):
            b_output_tensor = [b_output_tensor]
        if not isinstance(b_output_tensor_grad, list):
            b_output_tensor_grad = [b_output_tensor_grad]

        # Backward pass for loss function
        b_schedule_plan = b_output_tensor[0].schedule_plan
        b_output_tensor[0].schedule_plan = None
        if b_output_tensor_grad[0] is None and config.grad_scale_func is not None:
            # backward schedule plan
            loss_node = b_output_tensor[0].loss_func
            b_output_tensor[0].loss_func = None
            b_output_tensor[0] = config.grad_scale_func(b_output_tensor[0])
            torch.autograd.backward(b_output_tensor[0], grad_tensors=b_output_tensor_grad[0])
            b_output_tensor_grad[0] = loss_node.get_grad()

    grad = b_output_tensor_grad[0] if b_model else None
    with context_manager:  # autocast context
        # schedule forward and backward
        output_tensor = schedule_chunk_1f1b(
            f_schedule_plan,
            b_schedule_plan,
            grad,
            f_context=f_context,
            b_context=b_context,
            pre_forward=pre_forward,
            pre_backward=pre_backward,
            post_forward=post_forward,
            post_backward=post_backward,
        )

    # forward post process
    num_tokens = None
    if f_model is not None:
        with f_context:
            num_tokens = torch.tensor(0, dtype=torch.int)
            if parallel_state.is_pipeline_last_stage():
                if not collect_non_loss_data:
                    loss_node = ScheduleNode(
                        loss_func,
                        torch.cuda.current_stream(),
                        f_schedule_plan.event,
                        name="loss_func",
                    )
                    loss_func = loss_node.forward
                    outputs = loss_func(output_tensor)
                    if len(outputs) == 3:
                        output_tensor, num_tokens, loss_reduced = outputs
                        if not config.calculate_per_token_loss:
                            output_tensor /= num_tokens
                            output_tensor /= num_microbatches
                    else:
                        # preserve legacy loss averaging behavior
                        # (ie, over the number of microbatches)
                        assert len(outputs) == 2
                        output_tensor, loss_reduced = outputs
                        output_tensor = output_tensor / num_microbatches
                    forward_data_store.append(loss_reduced)

                    # attach loss_func on output_tensor
                    output_tensor.loss_func = loss_node
                else:
                    data = loss_func(output_tensor, non_loss_data=True)
                    forward_data_store.append(data)
            # attach schedule plan on output tensor
            output_tensor.schedule_plan = f_schedule_plan
            if config.timers is not None:
                config.timers('forward-compute').stop()

            # Set the loss scale for the auxiliary loss of the MoE layer.
            # Since we use a trick to do backward on the auxiliary loss, we need to set the scale
            # explicitly.
            if hasattr(config, 'num_moe_experts') and config.num_moe_experts is not None:
                # Calculate the loss scale based on the grad_scale_func if available,
                # else default to 1.
                loss_scale = (
                    config.grad_scale_func(torch.ones(1, device=output_tensor.device))
                    if config.grad_scale_func is not None
                    else torch.tensor(1.0)
                )
                # Set the loss scale
                MoEAuxLossAutoScaler.set_loss_scale(loss_scale / num_microbatches)

            if not unwrap_output_tensor:
                output_tensor, num_tokens = [output_tensor], num_tokens
    # backward post process
    input_tensor_grad = None
    if b_model is not None:
        input_tensor_grad = [None]
        if b_input_tensor is not None:
            input_tensor_grad = []
            for x in b_input_tensor:
                if x is None:
                    input_tensor_grad.append(None)
                else:
                    input_tensor_grad.append(x.grad)

        if unwrap_input_tensor_grad:
            input_tensor_grad = input_tensor_grad[0]

    return output_tensor, num_tokens, input_tensor_grad


def get_default_cls_for_unwrap():
    """Returns the default classes to unwrap from a model.

    This function provides a tuple of classes that should be unwrapped from a model
    to access the underlying GPTModel instance. It includes DistributedDataParallel
    and Float16Module by default, and also attempts to include LegacyFloat16Module
    if available for backward compatibility.

    Returns:
        tuple: A tuple of classes to unwrap from a model.
    """
    cls = (DistributedDataParallel, Float16Module)
    try:
        # legacy should not be used in core, but for backward compatibility, we support it here
        from megatron.legacy.model import Float16Module as LegacyFloat16Module

        cls = cls + (LegacyFloat16Module,)
    except:
        pass
    return cls


def unwrap_model(model, module_instances=get_default_cls_for_unwrap()):
    """Unwrap_model DistributedDataParallel and Float16Module wrapped model
    to return GPTModel instance
    """
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


class MemoryManagementStrategy:
    """Base class for memory management strategies.

    Different memory management strategies can be implemented by subclassing this class.
    These strategies control how tensors are handled in memory during the computation.
    """

    def handle_inputs(self, inputs, stream):
        """Process input tensors after computation.

        Args:
            inputs (tuple): Input tensors that have been used
            stream (torch.cuda.Stream): Current CUDA stream
        """
        pass

    def handle_outputs(self, outputs, stream):
        """Process output tensors after computation.

        Args:
            outputs (tuple): Output tensors produced by the computation
            stream (torch.cuda.Stream): Current CUDA stream
        """
        pass


class NoOpMemoryStrategy(MemoryManagementStrategy):
    """Strategy that performs no memory management operations.

    This is the default strategy - it doesn't free any memory.
    """

    pass


class FreeInputsMemoryStrategy(MemoryManagementStrategy):
    """Strategy that immediately frees input tensors after they are used.

    This strategy is useful for nodes where inputs are no longer needed
    after computation, helping to reduce memory usage.
    """

    def handle_inputs(self, inputs, stream):
        """Free input tensors by resizing their storage to zero.

        Args:
            inputs (tuple): Input tensors to be freed
            stream (torch.cuda.Stream): Current CUDA stream
        """
        for input in inputs:
            if input is not None:
                input.record_stream(stream)
                input.untyped_storage().resize_(0)
