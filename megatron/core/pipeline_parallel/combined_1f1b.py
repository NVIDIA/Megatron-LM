# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import contextlib
from contextlib import nullcontext
from typing import List, Union

import torch

from megatron.core.enums import Fp8Recipe
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.pipeline_parallel.utils import AbstractSchedulePlan, ScheduleNode
from megatron.core.utils import get_attr_wrapped_model

# Types
Shape = Union[List[int], torch.Size]


def schedule_chunk_1f1b(
    f_schedule_plan,
    b_schedule_plan,
    b_grad=None,
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
        b_grad=b_grad,
        f_context=f_context,
        b_context=b_context,
        pre_forward=pre_forward,
        pre_backward=pre_backward,
        post_forward=post_forward,
        post_backward=post_backward,
    )


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
    """Merged forward and backward step for overlap_moe_expert_parallel_comm.

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
    ), "checkpoint_activations_microbatch is not supported for overlap_moe_expert_parallel_comm"

    from .schedules import set_current_microbatch

    if f_model is not None and config.timers is not None:
        config.timers('forward-compute', log_level=2).start()

    if config.enable_autocast:
        context_manager = torch.autocast("cuda", dtype=config.autocast_dtype)
    else:
        context_manager = contextlib.nullcontext()

    # forward preprocess, the same as the forward_step()
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

            # GPTModel.build_schedule_plan(model_forward_inputs) is called in the forward_step_func.
            # The return value becomes (forward_schedule_plan, loss_function),
            # which is used to be (forward_output_tensor, loss_function).
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
        # The same as the backward_step()
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

        # Get the schedule plan from the output tensor
        b_schedule_plan = b_output_tensor[0].schedule_plan
        b_output_tensor[0].schedule_plan = None
        # Get the loss function from the output tensor
        loss_node = b_output_tensor[0].loss_func
        b_output_tensor[0].loss_func = None

        if b_output_tensor_grad[0] is None and config.grad_scale_func is not None:
            b_output_tensor[0] = config.grad_scale_func(b_output_tensor[0])
            # Backward pass for loss function
            torch.autograd.backward(b_output_tensor[0], grad_tensors=b_output_tensor_grad[0])
            b_output_tensor_grad[0] = loss_node.get_grad()

    # If fp8_recipe is delayed, wrap the entire pass with get_fp8_context(),
    # otherwise do nothing extra at the outer level
    # if we are using other fp8 recipes, then the context manager enter&exit are free
    # we can wrap fp8_context within the for loop over layers, so that we can fine-grained
    # control which layer will be fp8 or bf16
    use_outer_fp8_context = config.fp8 and config.fp8_recipe == Fp8Recipe.delayed
    outer_fp8_context = get_fp8_context(config) if use_outer_fp8_context else nullcontext()

    b_grad = b_output_tensor_grad[0] if b_model else None
    with context_manager and outer_fp8_context:  # autocast context and delayed fp8 context
        # schedule forward and backward
        output_tensor = schedule_chunk_1f1b(
            f_schedule_plan,
            b_schedule_plan,
            b_grad,
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
            from megatron.core.pipeline_parallel.schedules import forward_step_calc_loss

            loss_node = ScheduleNode(
                loss_func, torch.cuda.current_stream(), f_schedule_plan.event, name="loss_func"
            )
            loss_func = loss_node.forward
            output_tensor, num_tokens = forward_step_calc_loss(
                f_model,
                output_tensor,
                loss_func,
                config,
                f_context.vpp_rank,
                collect_non_loss_data,
                num_microbatches,
                forward_data_store,
            )
            # Set the schedule plan and loss function to the output tensor
            # This is used to get the schedule plan and loss function in the backward pass
            output_tensor.schedule_plan = f_schedule_plan
            output_tensor.loss_func = loss_node

            if not unwrap_output_tensor:
                output_tensor, num_tokens = [output_tensor], num_tokens
    # backward post process, the same as the backward_step()
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
