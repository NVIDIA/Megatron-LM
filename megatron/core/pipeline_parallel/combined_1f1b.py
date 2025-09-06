# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import contextlib
from contextlib import nullcontext
from typing import List, Union

import torch

from megatron.core.enums import Fp8Recipe
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.pipeline_parallel.utils import AbstractSchedulePlan, ScheduleNode, set_streams
from megatron.core.utils import get_attr_wrapped_model

# Types
Shape = Union[List[int], torch.Size]


def combined_1f1b_schedule_for_no_pipelining(
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    input_tensor,
    output_tensor_grad,
    forward_data_store,
    config,
    collect_non_loss_data,
    first_val_step,
    forward_only,
    no_sync_func,
    total_num_tokens,
    check_first_val_step,
):
    """Scheduler for 1f1b with no pipelining.

    This function schedules micro-batches in a way that the forward pass of Transformer layers
    for one micro-batch runs in parallel with the backward pass of another.
    Each layer's forward and backward operations are co-scheduled to maximize the overlap of
    their computations and communications.
    EP A2A in forward step is hidden by the attention/mlp computation in the backward step,
    and vice versa.
    Assuming we have 4 microbatches, the schedule is as follows:
    Phases 0: 1st microbatch forward
    Phases 1: 1st microbatch backward + 2nd microbatch forward
    Phases 2: 2nd microbatch backward + 3rd microbatch forward
    Phases 3: 3rd microbatch backward + 4th microbatch forward
    Phases 4: 4th microbatch backward
    """

    set_streams()
    # The forward step for the first microbatch is executed alone, no a2a overlapping
    output_tensor, num_tokens, _ = combined_forward_backward_step(
        forward_step_func,
        data_iterator,
        model,  # f_model
        num_microbatches,
        input_tensor,
        forward_data_store,
        None,  # b_model
        input_tensor,
        None,  # b_output_tensor
        None,  # b_output_tensor_grad
        config,
        collect_non_loss_data=collect_non_loss_data,
        checkpoint_activations_microbatch=None,
        is_first_microbatch=check_first_val_step(True),
        current_microbatch=0,
    )
    # The forward step is executed in parallel with the backward step of another microbatch
    # EP A2A in forward step is hidden by the attention/mlp computation in the backward step
    # Vice versa.
    with no_sync_func():
        for i in range(num_microbatches - 1):
            total_num_tokens += num_tokens
            output_tensor, num_tokens, _ = combined_forward_backward_step(
                forward_step_func,
                data_iterator,
                model,  # f_model
                num_microbatches,
                input_tensor,
                forward_data_store,
                model,  # b_model
                input_tensor,  # b_input_tensor
                output_tensor,  # b_output_tensor
                output_tensor_grad,  # b_output_tensor_grad
                config,
                collect_non_loss_data=collect_non_loss_data,
                checkpoint_activations_microbatch=None,
                is_first_microbatch=check_first_val_step((i + 1) == 0),
                current_microbatch=(i + 1),
            )
    total_num_tokens += num_tokens
    # The backward step for the last microbatch is executed alone, no a2a overlapping
    # Run computation for last microbatch out of context handler (want to synchronize gradients).
    output_tensor, num_tokens, _ = combined_forward_backward_step(
        forward_step_func,
        data_iterator,
        None,  # f_model
        num_microbatches,
        input_tensor,
        forward_data_store,
        model,  # b_model
        input_tensor,  # b_input_tensor
        output_tensor,  # b_output_tensor
        output_tensor_grad,  # b_output_tensor_grad
        config,
    )
    return forward_data_store, total_num_tokens


def combined_1f1b_schedule_for_interleaved_pipelining(
    config,
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    forward_data_store,
    forward_step_helper_preprocess,
    forward_step_helper_postprocess,
    backward_step_helper_preprocess,
    backward_step_helper_postprocess,
    get_microbatch_id_in_model_chunk,
    get_model_chunk_id,
    check_first_val_step,
    is_first_microbatch_for_model_chunk,
    collect_non_loss_data,
    f_virtual_microbatch_id=None,
    b_virtual_microbatch_id=None,
    pre_forward=None,
    pre_backward=None,
    post_forward=None,
    post_backward=None,
):
    """Helper method to run combined forward and backward step for A2A communication hiding.
    This method merges the functionality of `forward_step_helper` and `backward_step_helper` and
    eventually calls `combined_forward_backward_step` method defined in `combined_1f1b.py`.
    This method is called only if `overlap_moe_expert_parallel_comm` is true.

    Args:
        The arguments could be categorized into 2 groups:
        - Common arguments
          - f_virtual_microbatch_id, b_virtual_microbatch_id,
        - Arguments for combined_forward_backward_step()
          - config, forward_step_func, data_iterator, model, num_microbatches, forward_data_store
          - check_first_val_step, is_first_microbatch_for_model_chunk, collect_non_loss_data
          - pre_forward, pre_backward, post_forward, post_backward
        - Callables for the forward_step_helper() and backward_step_helper()
          - forward_step_helper_preprocess, forward_step_helper_postprocess
          - backward_step_helper_preprocess, backward_step_helper_postprocess
          - get_microbatch_id_in_model_chunk, get_model_chunk_id

    Returns:
        output_tensor (Tensor or list[Tensor]): The output object(s) from the forward step.
        input_tensor_grad (Tensor): The grad of the input tensor.

    Descriptions:
        This method merges the forward_step_helper() and backward_step_helper() in schedules.py.
        Assuming that:
            def forward_step_helper():
                # forward_step_helper_preprocess()
                # forward_step()
                # forward_step_helper_postprocess()
            def backward_step_helper():
                # backward_step_helper_preprocess()
                # backward_step()
                # backward_step_helper_postprocess()
        Then the combined_1f1b_schedule_for_interleaved_pipelining() method will be:
            def combined_1f1b_schedule_for_interleaved_pipelining():
                # forward_step_helper_preprocess()
                # backward_step_helper_preprocess()
                # combined_forward_backward_step() // merged forward_step() and backward_step()
                # forward_step_helper_postprocess()
                # backward_step_helper_postprocess()
    """

    set_streams()
    # forward prepare
    f_model_chunk_id = None
    f_microbatch_id = None
    input_tensor = None
    if f_virtual_microbatch_id is not None:
        f_microbatch_id = get_microbatch_id_in_model_chunk(f_virtual_microbatch_id, forward=True)
    if f_virtual_microbatch_id is not None:
        f_model_chunk_id = get_model_chunk_id(f_virtual_microbatch_id, forward=True)
        input_tensor = forward_step_helper_preprocess(
            f_virtual_microbatch_id, f_model_chunk_id, f_microbatch_id
        )
    # backward prepare
    b_model_chunk_id = None
    b_input_tensor = None
    b_output_tensor = None
    b_output_tensor_grad = None
    if b_virtual_microbatch_id is not None:
        b_model_chunk_id = get_model_chunk_id(b_virtual_microbatch_id, forward=False)
        b_input_tensor, b_output_tensor, b_output_tensor_grad = backward_step_helper_preprocess(
            b_virtual_microbatch_id, b_model_chunk_id
        )
    # Call combined forward and backward step to overlap the communication and computation
    output_tensor, num_tokens, input_tensor_grad = combined_forward_backward_step(
        forward_step_func,
        data_iterator[f_model_chunk_id] if f_model_chunk_id is not None else None,
        model[f_model_chunk_id] if f_model_chunk_id is not None else None,
        num_microbatches,
        input_tensor,
        forward_data_store,
        model[b_model_chunk_id] if b_model_chunk_id is not None else None,
        b_input_tensor,
        b_output_tensor,
        b_output_tensor_grad,
        config,
        f_model_chunk_id=f_model_chunk_id,
        pre_forward=pre_forward,
        pre_backward=pre_backward,
        post_forward=post_forward,
        post_backward=post_backward,
        collect_non_loss_data=collect_non_loss_data,
        checkpoint_activations_microbatch=None,
        is_first_microbatch=check_first_val_step(
            is_first_microbatch_for_model_chunk(f_virtual_microbatch_id)
            if f_virtual_microbatch_id is not None
            else None
        ),
        current_microbatch=f_microbatch_id,
    )
    # forward post process
    if f_model_chunk_id is not None:
        forward_step_helper_postprocess(f_model_chunk_id, output_tensor, num_tokens)
    # backward post process
    if b_model_chunk_id:
        # The same as the backward_step_helper
        backward_step_helper_postprocess(b_virtual_microbatch_id)
        if input_tensor is not None:
            assert input_tensor_grad is not None
    return output_tensor, input_tensor_grad


def combined_forward_backward_step(
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
    f_model_chunk_id=None,
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
    """Merged forward and backward step for combined 1f1b scheduler.

    Args:
        Need to accept the argument of both forward_step() and backward_step().
        forward_step_func (callable): A function returning a forward schedule plan which is
            an input of schedule_chunk_1f1b function.

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
        if is_first_microbatch and hasattr(f_model, 'set_is_first_microbatch'):
            f_model.set_is_first_microbatch()
        if current_microbatch is not None:
            set_current_microbatch(f_model, current_microbatch)
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
            unwrap_output_tensor = True

        set_input_tensor = get_attr_wrapped_model(f_model, "set_input_tensor")
        set_input_tensor(input_tensor)

    # build the schedule plan and get loss function for forward step
    if f_model is not None:
        # GPTModel.build_schedule_plan(model_forward_inputs) is called in the forward_step_func.
        # The return value becomes (forward_schedule_plan, loss_function),
        # which is used to be (forward_output_tensor, loss_function).
        with context_manager:  # autocast context
            unwrapped_model = get_attr_wrapped_model(
                f_model, "build_schedule_plan", return_model_obj=True
            )
            from megatron.core.models.gpt.gpt_model import GPTModel

            assert isinstance(unwrapped_model, GPTModel), (
                "The final unwrapped model must be a GPTModel instance "
                "since only GPTModel is supported for EP A2A overlapping."
            )
            f_schedule_plan, loss_func = forward_step_func(
                data_iterator, unwrapped_model, return_schedule_plan=True
            )
            assert isinstance(
                f_schedule_plan, AbstractSchedulePlan
            ), "first output of forward_step_func must be one instance of AbstractSchedulePlan"

    # backward preprocess, the same as the backward_step()
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

        # Get the schedule plan from the output tensor
        b_schedule_plan = b_output_tensor[0].schedule_plan
        b_output_tensor[0].schedule_plan = None
        # Get the loss function from the output tensor
        loss_node = b_output_tensor[0].loss_func
        b_output_tensor[0].loss_func = None

        if b_output_tensor_grad[0] is None:
            if config.grad_scale_func is not None:
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
    # combined forward and backward model chunk execution of two micro-batches
    with context_manager and outer_fp8_context:  # autocast context and delayed fp8 context
        # For GPT models, it calls common::TransformerModelChunkSchedulePlan.run(),
        output_tensor = type(f_schedule_plan or b_schedule_plan).run(
            f_schedule_plan,
            b_schedule_plan,
            b_grad=b_grad,
            pre_forward=pre_forward,
            pre_backward=pre_backward,
            post_forward=post_forward,
            post_backward=post_backward,
        )

    # forward post process
    num_tokens = None
    if f_model is not None:
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
            f_model_chunk_id,
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
