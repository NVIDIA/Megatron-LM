import contextlib
from typing import Iterator, List, Union

import torch

from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import (
    forward_step,
    backward_step,
    get_tensor_shapes,
    check_first_val_step,
    deallocate_output_tensor,
    recv_forward,
    send_forward,
    recv_backward,
    send_backward,
    send_forward_recv_backward,
    send_backward_recv_forward,
    clear_embedding_activation_buffer,
    bootstrap_and_profile_p2p_communication,
    finish_embedding_wgrad_compute,
)
from megatron.core.pipeline_parallel.schedule_timers import ScheduleTimers
from megatron.core.tensor_parallel.vocab_output_store import VocabOutputStore
from megatron.core.tensor_parallel.vocab_input_store import VocabInputStore
from megatron.core.utils import (
    get_model_config,
    get_model_type,
    get_model_xattn,
)
from megatron.training import get_args


def forward_backward_pipelining_with_interlaced_schedule(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: bool = None,
):
    """Run non-interleaved 1F1B schedule with language model head layer sharding
    on the vocabulary dimension, with communication between pipeline stages.

    Returns dictionary with losses if the last stage, empty dict otherwise."""

    assert isinstance(model, list)

    config = get_model_config(model[0])
    if config.overlap_p2p_comm:
        raise ValueError(
            "Non-interleaved pipeline parallelism does not support overlapping p2p communication"
        )
    
    assert not forward_only, "Vocab parallel is incompatible with forward only."

    # Needed only when gradients are finalized in M-Core
    if config.finalize_model_grads_func is not None and not forward_only:
        embedding_module = clear_embedding_activation_buffer(config, model[0])

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    # Increment iter_counter in ScheduleTimers
    ScheduleTimers.iter_counter += 1

    if ScheduleTimers.iter_counter == get_args().schedule_timer_end + 1:
        ScheduleTimers.sync_timer = False

    if ScheduleTimers.iter_counter == get_args().schedule_timer_end + 6:
        conclusion = ScheduleTimers.joint_conclusion(sync_timer=False, global_reduce=False)
        print(f"rank {torch.distributed.get_rank()} profiling conclusion: {conclusion}")

    run_timer = (
        get_args().schedule_timer_end + 5
        >= ScheduleTimers.iter_counter
        >= get_args().schedule_timer_start
    )

    # Compute number of warmup microbatches.
    num_warmup_microbatches = (
        (parallel_state.get_pipeline_model_parallel_world_size() * 3 // 2)
        - parallel_state.get_pipeline_model_parallel_rank()
    )
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
    first_stage_num_warmup_microbatches = min(
        (parallel_state.get_pipeline_model_parallel_world_size() * 3 // 2),
        num_microbatches,
    )
    num_cooldown_microbatches = (
        parallel_state.get_pipeline_model_parallel_world_size()
        - parallel_state.get_pipeline_model_parallel_rank()
    )
    num_cooldown_microbatches = min(num_cooldown_microbatches, num_microbatches)
    first_stage_num_cooldown_microbatches = min(
        parallel_state.get_pipeline_model_parallel_world_size(),
        num_microbatches,
    )
    num_waiting_microbatches = max(
        0,
        num_warmup_microbatches - num_cooldown_microbatches
        - (
            parallel_state.get_pipeline_model_parallel_world_size()
            - parallel_state.get_pipeline_model_parallel_rank()
            - 1
        ) // 2
    )
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches

    assert config.num_microbatches_with_partial_activation_checkpoints is None, 'not supported'

    model_type = get_model_type(model[0])
    encoder_decoder_xattn = get_model_xattn(model[0])

    rank = parallel_state.get_pipeline_model_parallel_rank()
    recv_tensor_shapes = get_tensor_shapes(
        rank=rank - 1,
        model_type=model_type,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        encoder_decoder_xattn=encoder_decoder_xattn,
    )
    send_tensor_shapes = get_tensor_shapes(
        rank=rank,
        model_type=model_type,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        encoder_decoder_xattn=encoder_decoder_xattn,
    )
    lm_head_tensor_shapes = get_tensor_shapes(
        rank=parallel_state.get_pipeline_model_parallel_world_size() - 1,
        model_type=model_type,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        encoder_decoder_xattn=encoder_decoder_xattn,
    )

    bootstrap_and_profile_p2p_communication(
        config, send_tensor_shapes, recv_tensor_shapes)

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    output_tensors = None
    total_num_tokens = torch.tensor(0, dtype=torch.int).cuda()

    input_tensors = [[], [], []]
    output_tensors = [[], [], []]
    forward_data_store = []

    # Storing grad output of the loss reduce stage from B step to the next F step.
    last_stage_forward_input_store = [None for _ in range(num_microbatches)]
    last_stage_output_store = None

    def receive_lm_head_input(microbatch_id):
        nonlocal config, num_microbatches, last_stage_forward_input_store, lm_head_tensor_shapes
        assert not config.variable_seq_lengths, 'not supported yet'
        if parallel_state.is_pipeline_last_stage():
            output_tensor = last_stage_forward_input_store[microbatch_id]
            torch.distributed.broadcast(
                output_tensor,
                parallel_state.get_pipeline_model_parallel_last_rank(),
                group=parallel_state.get_pipeline_model_parallel_group(),
            )
            last_stage_forward_input_store[microbatch_id] = None
        else:
            output_tensor = torch.empty(
                lm_head_tensor_shapes[0],
                dtype=config.pipeline_dtype,
                device=torch.cuda.current_device(),
                requires_grad=True,
            )
            torch.distributed.broadcast(
                output_tensor,
                parallel_state.get_pipeline_model_parallel_last_rank(),
                group=parallel_state.get_pipeline_model_parallel_group(),
            )

        return [output_tensor]

    def forward_step_helper(
        microbatch_id,
        input_tensor,
        run_timer
    ):
        """
        Executes forward step and completes language model head communication (if any). Returns
        the output tensor.

        Note: This function does not push the input and output tensors into `input_tensors` and
        `output_tensors`. The caller should do this after sending the output tensor.
        """
        nonlocal forward_step_func, data_iterator, model, num_microbatches, forward_data_store, \
                 config, collect_non_loss_data, encoder_decoder_xattn, total_num_tokens, forward_only, \
                 last_stage_output_store, first_val_step, forward_only

        if get_args().profile:
            torch.cuda.nvtx.range_push(f"F{microbatch_id}")
        
        parallel_state.set_virtual_vocab_parallel_chunk(0)

        if parallel_state.is_pipeline_first_stage():
            input_tensor = [None]

        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model[0],
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            None,
            check_first_val_step(first_val_step, forward_only, microbatch_id == 0),
            current_microbatch=microbatch_id,
            encoder_decoder_xattn=encoder_decoder_xattn,
            skip_loss_compute=True,
            run_timer=run_timer
        )

        total_num_tokens += num_tokens.item()

        if parallel_state.is_pipeline_last_stage():
            nonlocal last_stage_forward_input_store
            last_stage_forward_input_store[microbatch_id] = \
                output_tensor[0].clone().detach().to(config.pipeline_dtype).requires_grad_(True)

        if get_args().profile:
            torch.cuda.nvtx.range_pop()

        return output_tensor

    def backward_step_helper(
        microbatch_id,
        output_tensor_grad,
        run_timer,
    ):
        nonlocal input_tensors, output_tensors, num_microbatches, config, rank, enable_grad_sync, \
                 model_type, forward_step_func, data_iterator, model, forward_data_store, \
                 collect_non_loss_data, encoder_decoder_xattn, last_stage_output_store, \
                 first_val_step, forward_only

        if get_args().profile:
            torch.cuda.nvtx.range_push(f"B{microbatch_id}")

        # Enable grad sync for the last microbatch in the batch if the full
        # backward pass completes in the 1F1B stage.
        if microbatch_id == num_microbatches - 1:
            if config.grad_sync_func is None or rank == 0:
                enable_grad_sync()

        if parallel_state.is_pipeline_last_stage():
            output_tensor_grad = last_stage_output_store

        input_tensor = input_tensors[0].pop(0)
        output_tensor = output_tensors[0].pop(0)

        parallel_state.set_virtual_vocab_parallel_chunk(0)

        input_tensor_grad = backward_step(
            input_tensor, output_tensor, output_tensor_grad, model_type, config,
            run_timer=run_timer
        )

        if parallel_state.is_pipeline_first_stage():
            VocabInputStore.backward_store(input_tensor_grad[0])

        if get_args().profile:
            torch.cuda.nvtx.range_pop()

        return input_tensor_grad

    def lm_head_step_helper(
        microbatch_id,
        run_timer
    ):
        nonlocal input_tensors, output_tensors, model_type, config, num_microbatches, \
                 forward_step_func, data_iterator, model, forward_data_store, \
                 collect_non_loss_data, encoder_decoder_xattn, first_val_step, forward_only, \
                 last_stage_output_store

        input_tensor = receive_lm_head_input(microbatch_id)
        
        if get_args().profile:
            torch.cuda.nvtx.range_push(f"S{microbatch_id}")

        parallel_state.set_virtual_vocab_parallel_chunk(1)
        VocabOutputStore.microbatch_id = microbatch_id

        if run_timer:
            ScheduleTimers.for_chunk(0).s_cnt += 1
            ScheduleTimers.for_chunk(0).s.start()

        output_tensor, _ = forward_step(
            forward_step_func,
            data_iterator,
            model[1],
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            None,
            check_first_val_step(first_val_step, forward_only, microbatch_id == 0),
            current_microbatch=microbatch_id,
            encoder_decoder_xattn=encoder_decoder_xattn,
            force_loss_compute=True,
            run_timer=False
        )

        deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

        last_stage_output_store = backward_step(
            input_tensor, output_tensor, None, model_type, config,
            run_timer=False
        )

        if run_timer:
            ScheduleTimers.for_chunk(0).s.stop()

        if get_args().profile:
            torch.cuda.nvtx.range_pop()
    
    input_embedding_output_shape = None

    def input_embedding_forward_step_helper(
        microbatch_id,
    ):
        nonlocal forward_step_func, data_iterator, model, num_microbatches, forward_data_store, \
                 config, collect_non_loss_data, encoder_decoder_xattn, forward_only, first_val_step

        parallel_state.set_virtual_vocab_parallel_chunk(2)

        input_tensor = [None]

        if get_args().profile:
            torch.cuda.nvtx.range_push(f"IF{microbatch_id}")

        if run_timer:
            ScheduleTimers.for_chunk(0).input_f_cnt += 1
            ScheduleTimers.for_chunk(0).input_f.start()

        output_tensor, _ = forward_step(
            forward_step_func,
            data_iterator,
            model[2],
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            None,
            check_first_val_step(first_val_step, forward_only, microbatch_id == 0),
            current_microbatch=microbatch_id,
            encoder_decoder_xattn=encoder_decoder_xattn,
            skip_loss_compute=True,
            run_timer=False
        )

        if run_timer:
            ScheduleTimers.for_chunk(0).input_f.stop()
        
        if get_args().profile:
            torch.cuda.nvtx.range_pop()

        nonlocal input_embedding_output_shape
        input_embedding_output_shape = output_tensor[0].shape

        reduced_output_tensor = output_tensor[0].clone().detach().to(dtype=config.pipeline_dtype).requires_grad_(True)

        input_tensors[2].append(input_tensor)
        output_tensors[2].append(output_tensor)
        deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

        torch.distributed.all_reduce(
            reduced_output_tensor,
            torch.distributed.ReduceOp.SUM,
            group=parallel_state.get_lm_head_model_parallel_group(),
        )

        if parallel_state.is_pipeline_first_stage():
            VocabInputStore.forward_store(reduced_output_tensor, None)

        return

    def input_embedding_backward_step_helper(
        microbatch_id
    ):
        parallel_state.set_virtual_vocab_parallel_chunk(2)

        input_tensor = input_tensors[2].pop(0)
        output_tensor = output_tensors[2].pop(0)

        if parallel_state.is_pipeline_first_stage():
            output_tensor_grad = [VocabInputStore.backward_get()]
        else:
            output_tensor_grad = [
                torch.empty(
                    input_embedding_output_shape,
                    dtype=config.pipeline_dtype,
                    device=torch.cuda.current_device(),
                )
            ]

        torch.distributed.broadcast(
            output_tensor_grad[0],
            parallel_state.get_pipeline_model_parallel_first_rank(),
            group=parallel_state.get_lm_head_model_parallel_group(),
        )

        if get_args().profile:
            torch.cuda.nvtx.range_push(f"IB{microbatch_id}")

        if run_timer:
            ScheduleTimers.for_chunk(0).input_b_cnt += 1
            ScheduleTimers.for_chunk(0).input_b.start()

        backward_step(
            input_tensor, output_tensor, output_tensor_grad, model_type, config,
            run_timer=False
        )

        if run_timer:
            ScheduleTimers.for_chunk(0).input_b.stop()
        
        if get_args().profile:
            torch.cuda.nvtx.range_pop()
    
    def fused_input_embedding_step_helper(
        forward_microbatch_id, backward_microbatch_id,
    ):
        nonlocal forward_step_func, data_iterator, model, num_microbatches, forward_data_store, \
                 config, collect_non_loss_data, encoder_decoder_xattn, forward_only, first_val_step

        parallel_state.set_virtual_vocab_parallel_chunk(2)

        nonlocal input_embedding_output_shape

        if parallel_state.is_pipeline_first_stage():
            output_tensor_grad = [VocabInputStore.backward_get()]
        else:
            output_tensor_grad = [
                torch.empty(
                    input_embedding_output_shape,
                    dtype=config.pipeline_dtype,
                    device=torch.cuda.current_device(),
                )
            ]

        handle = torch.distributed.broadcast(
            output_tensor_grad[0],
            parallel_state.get_pipeline_model_parallel_first_rank(),
            group=parallel_state.get_lm_head_model_parallel_group(),
            async_op=True,
        )

        input_tensor = [None]

        if get_args().profile:
            torch.cuda.nvtx.range_push(f"IF{forward_microbatch_id}")

        if run_timer:
            ScheduleTimers.for_chunk(0).input_f_cnt += 1
            ScheduleTimers.for_chunk(0).input_f.start()

        output_tensor, _ = forward_step(
            forward_step_func,
            data_iterator,
            model[2],
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            None,
            check_first_val_step(first_val_step, forward_only, forward_microbatch_id == 0),
            current_microbatch=forward_microbatch_id,
            encoder_decoder_xattn=encoder_decoder_xattn,
            skip_loss_compute=True,
            run_timer=False
        )

        if run_timer:
            ScheduleTimers.for_chunk(0).input_f.stop()

        if get_args().profile:
            torch.cuda.nvtx.range_pop()

        reduced_output_tensor = output_tensor[0].clone().detach().to(dtype=config.pipeline_dtype).requires_grad_(True)

        input_tensors[2].append(input_tensor)
        output_tensors[2].append(output_tensor)
        deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

        handle.wait()

        handle = torch.distributed.all_reduce(
            reduced_output_tensor,
            torch.distributed.ReduceOp.SUM,
            group=parallel_state.get_lm_head_model_parallel_group(),
            async_op=True,
        )

        if parallel_state.is_pipeline_first_stage():
            VocabInputStore.forward_store(reduced_output_tensor, None)
        
        input_tensor = input_tensors[2].pop(0)
        output_tensor = output_tensors[2].pop(0)

        if get_args().profile:
            torch.cuda.nvtx.range_push(f"IB{backward_microbatch_id}")

        if run_timer:
            ScheduleTimers.for_chunk(0).input_b_cnt += 1
            ScheduleTimers.for_chunk(0).input_b.start()

        backward_step(
            input_tensor, output_tensor, output_tensor_grad, model_type, config,
            run_timer=False
        )

        if run_timer:
            ScheduleTimers.for_chunk(0).input_b.stop()
        
        if get_args().profile:
                torch.cuda.nvtx.range_pop()

        handle.wait()

        return

    assert not forward_only, "Not supported"

    num_input_embedding_forward_steps_remaining = num_microbatches
    num_input_embedding_backward_steps_remaining = num_microbatches

    for i in range(first_stage_num_warmup_microbatches - num_warmup_microbatches + 1):
        input_embedding_forward_step_helper(i)
        num_input_embedding_forward_steps_remaining -= 1

    # Run warmup forward passes.
    for i in range(num_warmup_microbatches):
        # F
        input_tensor = recv_forward(recv_tensor_shapes, config)

        input_embedding_forward_step_helper(
            num_microbatches - num_input_embedding_forward_steps_remaining
        )
        num_input_embedding_forward_steps_remaining -= 1
        output_tensor = forward_step_helper(
            i,
            input_tensor,
            run_timer,
        )
        # The communication for the last stage should be deferred until after the first S pass.
        if i < num_warmup_microbatches - 1:
            send_forward(output_tensor, send_tensor_shapes, config)
            input_tensors[0].append(input_tensor)
            output_tensors[0].append(output_tensor)
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)
    
    def execute_embedding_passes(microbatch_id):
        nonlocal num_microbatches, first_stage_num_cooldown_microbatches, \
                 num_input_embedding_backward_steps_remaining, \
                 num_input_embedding_forward_steps_remaining

        execute_embedding_forward = (
            (microbatch_id >= first_stage_num_cooldown_microbatches - 1)
            and (num_input_embedding_forward_steps_remaining > 0)
        )
        execute_embedding_backward = (
            (microbatch_id >= first_stage_num_cooldown_microbatches + 1)
            or (
                (not first_stage_is_fbs)
                and (microbatch_id == first_stage_num_cooldown_microbatches)
            )
        )

        if execute_embedding_forward and execute_embedding_backward:
            fused_input_embedding_step_helper(
                num_microbatches - num_input_embedding_forward_steps_remaining,
                num_microbatches - num_input_embedding_backward_steps_remaining,
            )
            num_input_embedding_forward_steps_remaining -= 1
            num_input_embedding_backward_steps_remaining -= 1
        elif execute_embedding_forward:
            input_embedding_forward_step_helper(
                num_microbatches - num_input_embedding_forward_steps_remaining,
            )
            num_input_embedding_forward_steps_remaining -= 1
        elif execute_embedding_backward:
            input_embedding_backward_step_helper(
                num_microbatches - num_input_embedding_backward_steps_remaining,
            )
            num_input_embedding_backward_steps_remaining -= 1

    for i in range(num_cooldown_microbatches):
        # S
        lm_head_step_helper(i, run_timer)
        if (i >= first_stage_num_cooldown_microbatches - 1) and (num_input_embedding_forward_steps_remaining > 0):
            input_embedding_forward_step_helper(
                num_microbatches - num_input_embedding_forward_steps_remaining
            )
            num_input_embedding_forward_steps_remaining -= 1
    
    is_fbs = (
        parallel_state.get_pipeline_model_parallel_rank() % 2
        == parallel_state.get_pipeline_model_parallel_world_size() % 2
    )
    first_stage_is_fbs = (
        parallel_state.get_pipeline_model_parallel_world_size() % 2
        == 0
    )

    for i in range(num_waiting_microbatches):
        # B S
        output_tensor_grad = recv_backward(send_tensor_shapes, config)

        if (i == num_waiting_microbatches - 1) and (is_fbs):
            send_forward(output_tensor, send_tensor_shapes, config)
            input_tensors[0].append(input_tensor)
            output_tensors[0].append(output_tensor)
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

        input_tensor_grad = backward_step_helper(
            i, output_tensor_grad, run_timer,
        )

        lm_head_step_helper(i + num_cooldown_microbatches, run_timer)
        execute_embedding_passes(i + num_cooldown_microbatches)

        if (i < num_waiting_microbatches - 1):
            send_backward(input_tensor_grad, recv_tensor_shapes, config)

    for i in range(num_microbatches_remaining):
        if is_fbs:
            # F B S
            output_tensor_grad = recv_backward(send_tensor_shapes, config)

            input_tensor = send_backward_recv_forward(
                input_tensor_grad, recv_tensor_shapes, config,
            )

            output_tensor = forward_step_helper(
                i + num_warmup_microbatches,
                input_tensor,
                run_timer,
            )
            reqs = send_forward(output_tensor, send_tensor_shapes, config, wait_on_reqs=False)

            input_tensor_grad = backward_step_helper(
                i + num_waiting_microbatches,
                output_tensor_grad,
                run_timer,
            )

            for req in reqs:
                req.wait()

            input_tensors[0].append(input_tensor)
            output_tensors[0].append(output_tensor)
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

            lm_head_step_helper(
                i + num_cooldown_microbatches + num_waiting_microbatches,
                run_timer
            )
            execute_embedding_passes(
                i + num_cooldown_microbatches + num_waiting_microbatches
            )
        else:
            # B F S
            send_backward(input_tensor_grad, recv_tensor_shapes, config)

            output_tensor_grad = send_forward_recv_backward(
                output_tensor, send_tensor_shapes, config,
            )
            input_tensors[0].append(input_tensor)
            output_tensors[0].append(output_tensor)
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

            input_tensor_grad = backward_step_helper(
                i + num_waiting_microbatches, output_tensor_grad, run_timer,
            )

            input_tensor = recv_forward(recv_tensor_shapes, config)
            output_tensor = forward_step_helper(
                i + num_warmup_microbatches,
                input_tensor,
                run_timer,
            )

            lm_head_step_helper(
                i + num_cooldown_microbatches + num_waiting_microbatches,
                run_timer
            )
            execute_embedding_passes(
                i + num_cooldown_microbatches + num_waiting_microbatches
            )
    
    final_f_sent = is_fbs

    for i in range(num_warmup_microbatches - num_cooldown_microbatches - num_waiting_microbatches):
        # B S
        if is_fbs:
            output_tensor_grad = recv_backward(send_tensor_shapes, config)

            send_backward(input_tensor_grad, recv_tensor_shapes, config)
        else:
            send_backward(input_tensor_grad, recv_tensor_shapes, config)

            if not final_f_sent:
                output_tensor_grad = send_forward_recv_backward(
                    output_tensor, send_tensor_shapes, config
                )
                input_tensors[0].append(input_tensor)
                output_tensors[0].append(output_tensor)
                deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)
                final_f_sent = True
            else:
                output_tensor_grad = recv_backward(send_tensor_shapes, config)

        input_tensor_grad = backward_step_helper(
            i + num_waiting_microbatches, output_tensor_grad, run_timer,
        )

        lm_head_step_helper(
            i + num_cooldown_microbatches + num_waiting_microbatches + num_microbatches_remaining,
            run_timer
        )
        if num_input_embedding_backward_steps_remaining > 0:
            input_embedding_backward_step_helper(
                num_microbatches - num_input_embedding_backward_steps_remaining
            )
            num_input_embedding_backward_steps_remaining -= 1

    for i in range(num_cooldown_microbatches):
        if is_fbs:
            output_tensor_grad = recv_backward(send_tensor_shapes, config)

            send_backward(input_tensor_grad, recv_tensor_shapes, config)
        else:
            send_backward(input_tensor_grad, recv_tensor_shapes, config)

            if not final_f_sent:
                output_tensor_grad = send_forward_recv_backward(
                    output_tensor, send_tensor_shapes, config
                )
                input_tensors[0].append(input_tensor)
                output_tensors[0].append(output_tensor)
                deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)
                final_f_sent = True
            else:
                output_tensor_grad = recv_backward(send_tensor_shapes, config)

        input_tensor_grad = backward_step_helper(
            i + num_microbatches - num_cooldown_microbatches, output_tensor_grad, run_timer,
        )

        if num_input_embedding_backward_steps_remaining > 0:
            input_embedding_backward_step_helper(
                num_microbatches - num_input_embedding_backward_steps_remaining
            )
            num_input_embedding_backward_steps_remaining -= 1
    
    send_backward(input_tensor_grad, recv_tensor_shapes, config)

    while num_input_embedding_backward_steps_remaining > 0:
        input_embedding_backward_step_helper(
            num_microbatches - num_input_embedding_backward_steps_remaining
        )
        num_input_embedding_backward_steps_remaining -= 1

    # Launch any remaining grad reductions.
    if no_sync_context is not None:
        enable_grad_sync()
        if config.grad_sync_func is not None:
            config.grad_sync_func(model.parameters())

    if config.finalize_model_grads_func is not None and not forward_only:

        # If defer_embedding_wgrad_compute is enabled we need to do the
        # weight gradient GEMM's here.
        finish_embedding_wgrad_compute(config, embedding_module)

        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
        config.finalize_model_grads_func(
            model, total_num_tokens if config.calculate_per_token_loss else None
        )

    if config.timers is not None:
        config.timers('forward-backward').stop()

    return forward_data_store

