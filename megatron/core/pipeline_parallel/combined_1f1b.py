# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import contextlib
from contextlib import nullcontext
from typing import List, Union

import torch

from megatron.core import parallel_state
from megatron.core.enums import Fp8Recipe
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.pipeline_parallel.utils import AbstractSchedulePlan, ScheduleNode
from megatron.core.transformer.moe.router import MoEAuxLossAutoScaler
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
            vp_stage = f_context.vpp_rank
            # The same as the forward_step()
            model_vp_stage = getattr(f_model, "vp_stage", None)
            if vp_stage is not None and model_vp_stage is not None:
                assert (
                    vp_stage == model_vp_stage
                ), f"vp_stage ({vp_stage}) doesn't match model_vp_stage ({model_vp_stage})"
            num_tokens = torch.tensor(0, dtype=torch.int)
            if parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage):
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
