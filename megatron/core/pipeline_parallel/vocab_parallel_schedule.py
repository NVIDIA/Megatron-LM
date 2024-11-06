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


LM_HEAD_RES_REDUCE_STREAM = None


def forward_backward_pipelining_with_vocab_parallel(
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
    """Run non-interleaved 1F1B schedule with Vocabulary Parallelism.

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
    
    if ScheduleTimers.iter_counter >= get_args().schedule_timer_end + 1:
        conclusion = ScheduleTimers.joint_conclusion()
        f = conclusion[0][0][0]
        b = conclusion[0][0][1]
        c = conclusion[0][0][3]
    else:
        f = 1
        b = 2
        c = 0
    
    assert f >= c, 'vocab parallel schedules assume f >= c, ' \
        'f < c will lead to additional pipeline bubbles due to incorrect ' \
        'placements for the S pass'
    assert b >= f, 'vocab parallel schedules assume b >= f for S pass placements'
    num_stages = parallel_state.get_pipeline_model_parallel_world_size()
    offset = 0
    is_bsf = [True]
    offsets = [0]
    while len(is_bsf) < num_stages:
        # we can either subtract f from the offset, or add (b - f) to the offset
        # FSM:
        # - BSF --[ - f ]--> BSF
        # - BSF --[ 0 ]--> BFS --[ (b - f) ]--> BSF
        if offset - f >= -b:
            offset = offset - f
            is_bsf.append(True)
            offsets.append(offset)
        else:
            is_bsf.append(False)
            offsets.append(offset)
            offset = offset + (b - f)
            is_bsf.append(True)
            offsets.append(offset)
    if len(is_bsf) > num_stages:
        is_bsf.pop()
        offsets.pop()
    
    is_bsf.reverse()
    offsets.reverse()

    num_warmup_s_pass = [0 for _ in range(num_stages)]
    for rank in range(num_stages - 2, -1, -1):
        if (not is_bsf[rank + 1]) and (is_bsf[rank]):
            num_warmup_s_pass[rank] = num_warmup_s_pass[rank + 1]
        else:
            num_warmup_s_pass[rank] = num_warmup_s_pass[rank + 1] + 1

    run_timer = (
        get_args().schedule_timer_end + 5
        >= ScheduleTimers.iter_counter
        >= get_args().schedule_timer_start
    )

    # Compute number of warmup microbatches.
    num_warmup_microbatches = (
        parallel_state.get_pipeline_model_parallel_world_size()
        - parallel_state.get_pipeline_model_parallel_rank()
    )
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
    first_stage_num_warmup_microbatches = min(
        parallel_state.get_pipeline_model_parallel_world_size(),
        num_microbatches,
    )
    num_microbatches_remaining = max(
        0,
        num_microbatches - num_warmup_microbatches - 1
    )
    if get_args().disable_backward_fusion:
        # Add one more warm-up microbatch.
        num_microbatches_remaining = max(0, num_microbatches_remaining - 1)

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

    global LM_HEAD_RES_REDUCE_STREAM
    LM_HEAD_RES_REDUCE_STREAM = torch.cuda.Stream()

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    output_tensors = None
    total_num_tokens = torch.tensor(0, dtype=torch.int).cuda()

    input_tensors = [[], [], []]
    output_tensors = [[], [], []]
    forward_data_store = []

    # Storing grad output of the loss reduce stage from B step to the next F step.
    last_stage_forward_input_store = None
    last_stage_backward_input_store = None
    lm_head_reduce_output_store = None

    comm_wait_tensor = torch.Tensor([0]).cuda()
    comm_wait_tensor.record_stream(LM_HEAD_RES_REDUCE_STREAM)

    def broadcast_lm_head_input(microbatch_id, output_tensor, grad_output):
        """
        Assumes `output_tensor` is retrieved from `last_stage_forward_input_store`.
        We do not store it into `last_stage_forward_input_store` again.
        """
        nonlocal config, last_stage_backward_input_store, num_microbatches
        assert parallel_state.is_pipeline_last_stage(), \
            "lm head input must be broadcasted from the last stage"
        assert not config.variable_seq_lengths, 'not supported yet'
        if microbatch_id == 0:
            broadcast_tensor = output_tensor[0].to(dtype=torch.float32)
        elif microbatch_id == num_microbatches:
            broadcast_tensor = grad_output[0].unsqueeze(-1)
        else:
            broadcast_tensor = torch.cat([output_tensor[0].to(dtype=torch.float32), \
                                          grad_output[0].unsqueeze(-1)], -1)

        torch.distributed.broadcast(
            broadcast_tensor,
            parallel_state.get_pipeline_model_parallel_last_rank(),
            group=parallel_state.get_lm_head_model_parallel_group(),
            async_op=True,
        )

        if microbatch_id > 0:
            last_stage_backward_input_store = grad_output[0]

    def receive_lm_head_input(microbatch_id):
        nonlocal config, num_microbatches, last_stage_forward_input_store, \
                 last_stage_backward_input_store, lm_head_tensor_shapes, \
                 lm_head_reduce_output_store

        if not parallel_state.is_pipeline_last_stage():
            last_dim_shape = 0
            if microbatch_id < num_microbatches:
                last_dim_shape += lm_head_tensor_shapes[0][-1]
            if microbatch_id > 0:
                last_dim_shape += 1

            broadcast_tensor = torch.empty(
                lm_head_tensor_shapes[0][:-1] + (last_dim_shape,),
                dtype=torch.float32,
                device=torch.cuda.current_device(),
                requires_grad=True,
            )

            handle = torch.distributed.broadcast(
                broadcast_tensor,
                parallel_state.get_pipeline_model_parallel_last_rank(),
                group=parallel_state.get_lm_head_model_parallel_group(),
                async_op=True,
            )

        def callback():
            nonlocal broadcast_tensor, handle, microbatch_id, num_microbatches, \
                     config, last_stage_forward_input_store, last_stage_backward_input_store, \
                     lm_head_tensor_shapes, lm_head_reduce_output_store
            
            if not parallel_state.is_pipeline_last_stage():
                handle.wait()

            if microbatch_id < num_microbatches:
                if parallel_state.is_pipeline_last_stage():
                    output_tensor = last_stage_forward_input_store
                    last_stage_forward_input_store = None
                else:
                    output_tensor = broadcast_tensor[:, :, :lm_head_tensor_shapes[0][-1]].clone().to(dtype=config.pipeline_dtype)
            else:
                output_tensor = None
            
            if microbatch_id > 0:
                # Ensure that the reduction is complete.
                global LM_HEAD_RES_REDUCE_STREAM
                torch.cuda.current_stream().wait_stream(LM_HEAD_RES_REDUCE_STREAM)
                logits_max, sum_exp_logits, _, _ = lm_head_reduce_output_store

                if parallel_state.is_pipeline_last_stage():
                    grad_output = last_stage_backward_input_store
                    last_stage_backward_input_store = None
                else:
                    grad_output = broadcast_tensor[:, :, -1]

                if config.sequence_parallel:
                    gathered_tensor_shape = list(sum_exp_logits.shape)
                    gathered_tensor_shape[0] *= parallel_state.get_tensor_model_parallel_world_size()
                    sum_exp_logits_buffer = torch.empty(
                        gathered_tensor_shape,
                        dtype=sum_exp_logits.dtype,
                        device=torch.cuda.current_device(),
                    )
                    torch.distributed.all_gather_into_tensor(
                        sum_exp_logits_buffer,
                        sum_exp_logits.contiguous(),
                        group=parallel_state.get_tensor_model_parallel_group(),
                    )
                    sum_exp_logits = sum_exp_logits_buffer
                    logits_max_buffer = torch.empty(
                        gathered_tensor_shape,
                        dtype=logits_max.dtype,
                        device=torch.cuda.current_device(),
                    )
                    torch.distributed.all_gather_into_tensor(
                        logits_max_buffer,
                        logits_max.contiguous(),
                        group=parallel_state.get_tensor_model_parallel_group(),
                    )
                    logits_max = logits_max_buffer
                    grad_output_buffer = torch.empty(
                        gathered_tensor_shape,
                        dtype=grad_output.dtype,
                        device=torch.cuda.current_device(),
                    )
                    torch.distributed.all_gather_into_tensor(
                        grad_output_buffer,
                        grad_output.contiguous(),
                        group=parallel_state.get_tensor_model_parallel_group(),
                    )
                    grad_output = grad_output_buffer
            else:
                sum_exp_logits = None
                logits_max = None
                grad_output = None

            return [output_tensor], sum_exp_logits, logits_max, [grad_output]
        
        return callback

    def sequence_shard(t: torch.Tensor, *, dim: int = 0):
        nonlocal config
        if not config.sequence_parallel:
            return t
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        rank = parallel_state.get_tensor_model_parallel_rank()
        dim_size = t.size(dim=dim) // world_size
        slices = [slice(None)] * t.dim()
        slices[dim] = slice(rank * dim_size, (rank + 1) * dim_size)
        return t[tuple(slices)]
    
    def reduce_lm_head_res_alg1(microbatch_id, logits_max, sum_exp_logits, predicted_logits, target_mask, grad_input):
        """
        Reduces `logits_max`, `sum_exp_logits`, `predicted_logits` and
        `grad_input` among all pipeline parallel ranks.
        """
        global LM_HEAD_RES_REDUCE_STREAM

        if microbatch_id < num_microbatches:
            logits_max = sequence_shard(logits_max)
            sum_exp_logits = sequence_shard(sum_exp_logits)
            predicted_logits = sequence_shard(predicted_logits)
            target_mask = sequence_shard(target_mask)

            for tensor in (logits_max, sum_exp_logits, predicted_logits, target_mask):
                tensor.record_stream(LM_HEAD_RES_REDUCE_STREAM)

            LM_HEAD_RES_REDUCE_STREAM.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(LM_HEAD_RES_REDUCE_STREAM):
                local_logits_max = logits_max.clone()
                handle = torch.distributed.all_reduce(
                    logits_max,
                    torch.distributed.ReduceOp.MAX,
                    group=parallel_state.get_lm_head_model_parallel_group(),
                    async_op=True,
                )
                handle.wait()
                local_logits_max -= logits_max

                predicted_logits += local_logits_max
                predicted_logits[target_mask] = 0.0
                handle = torch.distributed.all_reduce(
                    predicted_logits,
                    torch.distributed.ReduceOp.SUM,
                    group=parallel_state.get_lm_head_model_parallel_group(),
                    async_op=True,
                )
                handle.wait()

                local_logits_max.exp_()
                sum_exp_logits.mul_(local_logits_max)
                handle = torch.distributed.all_reduce(
                    sum_exp_logits,
                    torch.distributed.ReduceOp.SUM,
                    group=parallel_state.get_lm_head_model_parallel_group(),
                    async_op=True,
                )
                handle.wait()
        
        if microbatch_id > 0:
            grad_input.record_stream(LM_HEAD_RES_REDUCE_STREAM)
            LM_HEAD_RES_REDUCE_STREAM.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(LM_HEAD_RES_REDUCE_STREAM):
                handle = torch.distributed.all_reduce(
                    grad_input,
                    torch.distributed.ReduceOp.SUM,
                    group=parallel_state.get_lm_head_model_parallel_group(),
                    async_op=True,
                )
                handle.wait()

        return logits_max, sum_exp_logits, predicted_logits, grad_input

    def reduce_lm_head_res_alg2(logits_max, sum_exp_logits, predicted_logits, target_mask, softmax_grad_input, ground_truth_grad_input):
        """
        Reduces `logits_max`, `sum_exp_logits`, `predicted_logits` and
        `grad_input` among all pipeline parallel ranks.
        """

        logits_max = sequence_shard(logits_max)
        sum_exp_logits = sequence_shard(sum_exp_logits)
        predicted_logits = sequence_shard(predicted_logits)
        target_mask = sequence_shard(target_mask)

        global LM_HEAD_RES_REDUCE_STREAM

        for tensor in (logits_max, sum_exp_logits, predicted_logits, target_mask, softmax_grad_input,
                       ground_truth_grad_input):
            tensor.record_stream(LM_HEAD_RES_REDUCE_STREAM)

        LM_HEAD_RES_REDUCE_STREAM.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(LM_HEAD_RES_REDUCE_STREAM):
            local_logits_max = logits_max.clone()
            handle = torch.distributed.all_reduce(
                logits_max,
                torch.distributed.ReduceOp.MAX,
                group=parallel_state.get_lm_head_model_parallel_group(),
                async_op=True,
            )
            handle.wait()
            local_logits_max -= logits_max

            predicted_logits += local_logits_max
            predicted_logits[target_mask] = 0.0
            handle = torch.distributed.all_reduce(
                predicted_logits,
                torch.distributed.ReduceOp.SUM,
                group=parallel_state.get_lm_head_model_parallel_group(),
                async_op=True,
            )
            handle.wait()

            local_logits_max.exp_()
            sum_exp_logits.mul_(local_logits_max)
            local_sum_exp_logits = sum_exp_logits.clone()
            handle = torch.distributed.all_reduce(
                sum_exp_logits,
                torch.distributed.ReduceOp.SUM,
                group=parallel_state.get_lm_head_model_parallel_group(),
                async_op=True,
            )
            handle.wait()

            local_sum_exp_logits.div_(sum_exp_logits)
            softmax_grad_input.mul_(local_sum_exp_logits.unsqueeze(-1))
            softmax_grad_input -= ground_truth_grad_input
            handle = torch.distributed.all_reduce(
                softmax_grad_input,
                torch.distributed.ReduceOp.SUM,
                group=parallel_state.get_lm_head_model_parallel_group(),
                async_op=True,
            )
            handle.wait()

        return logits_max, sum_exp_logits, predicted_logits, softmax_grad_input

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
                 first_val_step, forward_only
        
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
            last_stage_forward_input_store = output_tensor[0].clone().detach() \
                                             .to(config.pipeline_dtype).requires_grad_(True)
            if microbatch_id == 0:
                broadcast_lm_head_input(microbatch_id, [last_stage_forward_input_store], None)
        
        if get_args().profile:
            torch.cuda.nvtx.range_pop()

        return output_tensor

    input_embedding_backward_callback = lambda: None

    def loss_calculation_helper(
        microbatch_id,
    ):
        assert parallel_state.is_pipeline_last_stage(), 'loss is only calculated at' \
            'the last pipeline parallel stage'
        nonlocal lm_head_reduce_output_store, num_microbatches, config, \
                 model_type, forward_step_func, data_iterator, model, forward_data_store, \
                 collect_non_loss_data, encoder_decoder_xattn, lm_head_reduce_output_store, \
                 first_val_step, forward_only, rank
        
        # Ensure that the reduction is complete.
        global LM_HEAD_RES_REDUCE_STREAM
        torch.cuda.current_stream().wait_stream(LM_HEAD_RES_REDUCE_STREAM)

        _, sum_exp_logits, predicted_logits, _ = lm_head_reduce_output_store

        # Calculate the loss. Then, execute the function that reduces the losses.

        input_tensor = torch.log(sum_exp_logits) - predicted_logits

        if config.sequence_parallel:
            gathered_tensor_shapes = list(lm_head_tensor_shapes[0][:-1])
            gathered_tensor_shapes[0] *= parallel_state.get_tensor_model_parallel_world_size()
            input_tensor_buffer = torch.empty(
                gathered_tensor_shapes,
                dtype=input_tensor.dtype,
                device=torch.cuda.current_device()
            )
            torch.distributed.all_gather_into_tensor(
                input_tensor_buffer,
                input_tensor,
                group=parallel_state.get_tensor_model_parallel_group(),
            )
            input_tensor = input_tensor_buffer

        input_tensor = [input_tensor.clone().detach().requires_grad_(True)]

        parallel_state.set_virtual_vocab_parallel_chunk(3)

        output_tensor, _ = forward_step(
            forward_step_func,
            data_iterator,
            model[3],
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            None,
            check_first_val_step(first_val_step, forward_only, microbatch_id == 0),
            current_microbatch=microbatch_id,
            encoder_decoder_xattn=encoder_decoder_xattn,
            run_timer=False
        )

        output_tensor_grad = backward_step(
            input_tensor, output_tensor, [None], model_type, config,
            run_timer=False
        )

        if get_args().disable_backward_fusion:
            if microbatch_id < num_microbatches:
                nonlocal last_stage_forward_input_store
                broadcast_lm_head_input(microbatch_id + 1, [last_stage_forward_input_store],
                                        [sequence_shard(output_tensor_grad[0])])
            else:
                broadcast_lm_head_input(microbatch_id + 1, None,
                                        [sequence_shard(output_tensor_grad[0])])
        else:
            return output_tensor_grad

    def backward_step_helper(
        microbatch_id,
        output_tensor_grad,
        run_timer,
    ):
        nonlocal input_tensors, output_tensors, num_microbatches, config, rank, enable_grad_sync, \
                 model_type, forward_step_func, data_iterator, model, forward_data_store, \
                 collect_non_loss_data, encoder_decoder_xattn, lm_head_reduce_output_store, \
                 first_val_step, forward_only
        
        post_process = lambda: None

        if get_args().profile:
            torch.cuda.nvtx.range_push(f"B{microbatch_id}")

        # Enable grad sync for the last microbatch in the batch if the full
        # backward pass completes in the 1F1B stage.
        if microbatch_id == num_microbatches - 1:
            if config.grad_sync_func is None or rank == 0:
                enable_grad_sync()

        if parallel_state.is_pipeline_last_stage():
            # Ensure that the reduction is complete.
            global LM_HEAD_RES_REDUCE_STREAM
            torch.cuda.current_stream().wait_stream(LM_HEAD_RES_REDUCE_STREAM)

            _, _, _, grad_input = lm_head_reduce_output_store

            if not get_args().disable_backward_fusion:
                output_tensor_grad = loss_calculation_helper(microbatch_id)

                if microbatch_id < num_microbatches - 1:
                    nonlocal last_stage_forward_input_store
                    broadcast_lm_head_input(microbatch_id + 1, [last_stage_forward_input_store],
                                            [sequence_shard(output_tensor_grad[0])])
                else:
                    output_tensor_grad_store = output_tensor_grad
                    post_process = lambda: broadcast_lm_head_input(microbatch_id + 1, None,
                                                                [sequence_shard(output_tensor_grad_store[0])])

            # Calculate the input grads of the lm head layer, without calling backward.
            input_tensor_grad = [grad_input]

            if not get_args().disable_backward_fusion:
                input_tensor_grad[0].mul_(sequence_shard(output_tensor_grad[0]).unsqueeze(dim=-1))

            output_tensor_grad = input_tensor_grad

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

        return input_tensor_grad, post_process

    def lm_head_step_helper(
        microbatch_id,
        lm_head_inputs,
        run_timer
    ):
        nonlocal input_tensors, output_tensors, model_type, config, num_microbatches, \
                 forward_step_func, data_iterator, model, forward_data_store, \
                 collect_non_loss_data, encoder_decoder_xattn, first_val_step, forward_only

        if get_args().profile:
            torch.cuda.nvtx.range_push(f"S{microbatch_id}")

        lm_head_input_tensor, sum_exp_logits, logits_max, grad_output = lm_head_inputs

        parallel_state.set_virtual_vocab_parallel_chunk(1)
        VocabOutputStore.microbatch_id = microbatch_id

        if (run_timer) and (0 < microbatch_id < num_microbatches):
            ScheduleTimers.for_chunk(0).s_cnt += 1
            ScheduleTimers.for_chunk(0).s.start()

        if microbatch_id > 0:
            input_tensor = input_tensors[1].pop(0)
            output_tensor = output_tensors[1].pop(0)

            # Only for weight grad updates, input grad returned is ignored.
            VocabOutputStore.backward_store(sum_exp_logits, logits_max, grad_output[0])
            grad_input = backward_step(
                input_tensor, output_tensor, [grad_output[0].transpose(0, 1)], model_type, config,
                run_timer=False
            )
        else:
            grad_input = [None]

        if microbatch_id < num_microbatches:
            output_tensor, _ = forward_step(
                forward_step_func,
                data_iterator,
                model[1],
                num_microbatches,
                lm_head_input_tensor,
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
            output_tensor = [output_tensor[0].clone()]
            sum_exp_logits, logits_max, predicted_logits, target_mask, softmax_grad_input, \
                ground_truth_grad_input = VocabOutputStore.forward_get()

            input_tensors[1].append(lm_head_input_tensor)
            output_tensors[1].append(output_tensor)
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

            if get_args().disable_backward_fusion:
                lm_head_res = (logits_max, sum_exp_logits, predicted_logits, target_mask,
                               grad_input[0])
            else:
                lm_head_res = (logits_max, sum_exp_logits, predicted_logits, target_mask,
                           softmax_grad_input, ground_truth_grad_input)
        else:
            if get_args().disable_backward_fusion:
                lm_head_res = (None, None, None, None, grad_input[0])
            else:
                lm_head_res = None
        
        if (run_timer) and (0 < microbatch_id < num_microbatches):
            ScheduleTimers.for_chunk(0).s.stop()

        if get_args().profile:
            torch.cuda.nvtx.range_pop()

        return lm_head_res

    input_embedding_output_shape = None
    
    def input_embedding_forward_step_helper(
        microbatch_id,
    ):
        nonlocal forward_step_func, data_iterator, model, num_microbatches, forward_data_store, \
                 config, collect_non_loss_data, encoder_decoder_xattn, forward_only, first_val_step, \
                 run_timer
        
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

        def callback():
            nonlocal reduced_output_tensor

            torch.distributed.all_reduce(
                comm_wait_tensor,
                torch.distributed.ReduceOp.MAX,
                group=parallel_state.get_lm_head_model_parallel_group(),
                async_op=True,
            )

            handle = torch.distributed.all_reduce(
                reduced_output_tensor,
                torch.distributed.ReduceOp.SUM,
                group=parallel_state.get_lm_head_model_parallel_group(),
                async_op=True,
            )

            if parallel_state.is_pipeline_first_stage():
                VocabInputStore.forward_store(reduced_output_tensor, handle)

            return
        
        return callback
    
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

        torch.distributed.all_reduce(
                comm_wait_tensor,
                torch.distributed.ReduceOp.MAX,
                group=parallel_state.get_lm_head_model_parallel_group(),
                async_op=True,
            )

        handle = torch.distributed.broadcast(
            output_tensor_grad[0],
            parallel_state.get_pipeline_model_parallel_first_rank(),
            group=parallel_state.get_lm_head_model_parallel_group(),
            async_op=True,
        )

        def callback():
            nonlocal input_tensor, output_tensor,output_tensor_grad, model_type, \
                     config, handle, run_timer

            handle.wait()

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

        return callback

    assert not forward_only, "Not supported"

    num_input_embedding_forward_steps_remaining = num_microbatches
    num_input_embedding_backward_steps_remaining = num_microbatches

    for i in range(first_stage_num_warmup_microbatches - num_warmup_microbatches + 1):
        input_embedding_forward_step_helper(i)()
        num_input_embedding_forward_steps_remaining -= 1

    # Run warmup forward passes.
    for i in range(num_warmup_microbatches):
        # Decide to checkpoint all layers' activations of the current micro-batch
        input_tensor = recv_forward(recv_tensor_shapes, config)

        input_embedding_forward_step_helper(
            num_microbatches - num_input_embedding_forward_steps_remaining
        )()
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

    num_remaining_s_pass = num_microbatches
    lm_head_inputs = receive_lm_head_input(0)()
    lm_head_res = lm_head_step_helper(0, lm_head_inputs, run_timer)
    num_remaining_s_pass -= 1

    if get_args().disable_backward_fusion:
        lm_head_res = reduce_lm_head_res_alg1(0, *lm_head_res)
    else:
        lm_head_res = reduce_lm_head_res_alg2(*lm_head_res)
    lm_head_reduce_output_store = lm_head_res

    if get_args().disable_backward_fusion:
        input_embedding_forward_step_helper(
            num_microbatches - num_input_embedding_forward_steps_remaining
        )()
        num_input_embedding_forward_steps_remaining -= 1

    if num_warmup_microbatches > 0:
        send_forward(output_tensor, send_tensor_shapes, config)
        input_tensors[0].append(input_tensor)
        output_tensors[0].append(output_tensor)
        deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

    if num_warmup_microbatches + 1 <= num_microbatches:
        # Decide to checkpoint all layers' activations of the current micro-batch
        input_tensor = recv_forward(recv_tensor_shapes, config)

    if num_warmup_microbatches + 1 <= num_microbatches:
        output_tensor = forward_step_helper(
            num_warmup_microbatches,
            input_tensor,
            run_timer,
        )
        if (not get_args().disable_backward_fusion) and (parallel_state.is_pipeline_last_stage()):
            output_tensor_grad = send_forward_recv_backward(
                output_tensor, send_tensor_shapes, config
            )
            input_tensors[0].append(input_tensor)
            output_tensors[0].append(output_tensor)
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)
    
    if get_args().disable_backward_fusion:
        if parallel_state.is_pipeline_last_stage():
            loss_calculation_helper(0)

        lm_head_inputs = receive_lm_head_input(1)()
        lm_head_res = lm_head_step_helper(1, lm_head_inputs, run_timer)
        num_remaining_s_pass -= 1

        lm_head_res = reduce_lm_head_res_alg1(1, *lm_head_res)
        lm_head_reduce_output_store = lm_head_res

        if num_warmup_microbatches + 1 <= num_microbatches:
            send_forward(output_tensor, send_tensor_shapes, config)
            input_tensors[0].append(input_tensor)
            output_tensors[0].append(output_tensor)
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

        if num_warmup_microbatches + 2 <= num_microbatches:
            input_tensor = recv_forward(recv_tensor_shapes, config)

        if num_warmup_microbatches + 2 <= num_microbatches:
            output_tensor = forward_step_helper(
                num_warmup_microbatches + 1,
                input_tensor,
                run_timer,
            )
            if parallel_state.is_pipeline_last_stage():
                output_tensor_grad = send_forward_recv_backward(
                    output_tensor, send_tensor_shapes, config
                )
                input_tensors[0].append(input_tensor)
                output_tensors[0].append(output_tensor)
                deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

    num_warmup_s_pass_rank = num_warmup_s_pass[
        parallel_state.get_pipeline_model_parallel_rank()
    ]

    if get_args().disable_backward_fusion:
        warmup_offset = 1
    else:
        warmup_offset = 0
    
    for i in range(num_warmup_s_pass_rank):
        if num_remaining_s_pass >= 1 - warmup_offset:
            lm_head_inputs = receive_lm_head_input(i + 1 + warmup_offset)()
            lm_head_res = lm_head_step_helper(i + 1 + warmup_offset, lm_head_inputs, run_timer)
            if (i + 1 >= num_warmup_s_pass[0]) and (num_input_embedding_forward_steps_remaining > 0):
                input_embedding_forward_callback = input_embedding_forward_step_helper(
                    num_microbatches - num_input_embedding_forward_steps_remaining
                )
                num_input_embedding_forward_steps_remaining -= 1
            else:
                input_embedding_forward_callback = lambda: None
            if (
                (parallel_state.get_pipeline_model_parallel_rank() == parallel_state.get_pipeline_model_parallel_world_size() - 2)
                or (not is_bsf[parallel_state.get_pipeline_model_parallel_rank() + 1])
            ):
                input_embedding_forward_callback()
                if get_args().disable_backward_fusion:
                    lm_head_reduce_output_store = reduce_lm_head_res_alg1(i + 2, *lm_head_res)
                else:
                    lm_head_reduce_output_store = reduce_lm_head_res_alg2(*lm_head_res)
            if i == num_warmup_s_pass_rank - 1:
                output_tensor_grad = send_forward_recv_backward(
                    output_tensor, send_tensor_shapes, config
                )
                input_tensors[0].append(input_tensor)
                output_tensors[0].append(output_tensor)
                deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)
            if (
                (parallel_state.get_pipeline_model_parallel_rank() != parallel_state.get_pipeline_model_parallel_world_size() - 2)
                and (is_bsf[parallel_state.get_pipeline_model_parallel_rank() + 1])
            ):
                input_embedding_forward_callback()
                if get_args().disable_backward_fusion:
                    lm_head_reduce_output_store = reduce_lm_head_res_alg1(i + 2, *lm_head_res)
                else:
                    lm_head_reduce_output_store = reduce_lm_head_res_alg2(*lm_head_res)
            num_remaining_s_pass -= 1

    # Run 1F1B in steady state.
    for i in range(num_microbatches_remaining):
        if (
            (is_bsf[parallel_state.get_pipeline_model_parallel_rank()])
            or (offsets[parallel_state.get_pipeline_model_parallel_rank()] - f > -b)
        ):
            if num_microbatches - num_remaining_s_pass >= num_warmup_s_pass[0] + 2 + warmup_offset:
                input_embedding_backward_callback = input_embedding_backward_step_helper(
                    num_microbatches - num_input_embedding_backward_steps_remaining
                )
                num_input_embedding_backward_steps_remaining -= 1
            else:
                input_embedding_backward_callback = lambda: None
            receive_lm_head_input_callback = receive_lm_head_input(num_microbatches - num_remaining_s_pass)
        
        if get_args().disable_backward_fusion and parallel_state.is_pipeline_last_stage():
            loss_calculation_helper(i + 1)

        input_tensor_grad, _ = backward_step_helper(
            i, output_tensor_grad, run_timer,
        )

        if is_bsf[parallel_state.get_pipeline_model_parallel_rank()]:
            lm_head_inputs = receive_lm_head_input_callback()
            lm_head_res = lm_head_step_helper(num_microbatches - num_remaining_s_pass, lm_head_inputs, run_timer)
            if (
                (num_microbatches - num_remaining_s_pass >= num_warmup_s_pass[0] + warmup_offset)
                and (num_input_embedding_forward_steps_remaining > 0)
            ):
                input_embedding_forward_callback = input_embedding_forward_step_helper(
                    num_microbatches - num_input_embedding_forward_steps_remaining
                )
                num_input_embedding_forward_steps_remaining -= 1
            else:
                input_embedding_forward_callback = lambda: None

            input_embedding_backward_callback()
            num_remaining_s_pass -= 1

        if not parallel_state.is_pipeline_last_stage():
            input_tensor = send_backward_recv_forward(
                input_tensor_grad, recv_tensor_shapes, config
            )

        if (
            (is_bsf[parallel_state.get_pipeline_model_parallel_rank()])
            and (offsets[parallel_state.get_pipeline_model_parallel_rank()] + f >= 0)
        ):
            input_embedding_forward_callback()
            if get_args().disable_backward_fusion:
                lm_head_reduce_output_store = reduce_lm_head_res_alg1(
                    num_microbatches - num_remaining_s_pass - 1, *lm_head_res
                )
            else:
                lm_head_reduce_output_store = reduce_lm_head_res_alg2(*lm_head_res)

        if parallel_state.is_pipeline_last_stage():
            input_tensor = send_backward_recv_forward(
                input_tensor_grad, recv_tensor_shapes, config
            )

        if (
            (not is_bsf[parallel_state.get_pipeline_model_parallel_rank()])
            and (offsets[parallel_state.get_pipeline_model_parallel_rank()] - f <= -b)
        ):
            if num_microbatches - num_remaining_s_pass >= num_warmup_s_pass[0] + 2 + warmup_offset:
                input_embedding_backward_callback = input_embedding_backward_step_helper(
                    num_microbatches - num_input_embedding_backward_steps_remaining
                )
                num_input_embedding_backward_steps_remaining -= 1
            else:
                input_embedding_backward_callback = lambda: None
            receive_lm_head_input_callback = receive_lm_head_input(num_microbatches - num_remaining_s_pass)

        output_tensor = forward_step_helper(
            i + num_warmup_microbatches + 1 + warmup_offset,
            input_tensor,
            run_timer,
        )

        if not is_bsf[parallel_state.get_pipeline_model_parallel_rank()]:
            lm_head_inputs = receive_lm_head_input_callback()
            lm_head_res = lm_head_step_helper(num_microbatches - num_remaining_s_pass, lm_head_inputs, run_timer)
            if (
                (num_microbatches - num_remaining_s_pass >= num_warmup_s_pass[0] + warmup_offset)
                and (num_input_embedding_forward_steps_remaining > 0)
            ):
                input_embedding_forward_callback = input_embedding_forward_step_helper(
                    num_microbatches - num_input_embedding_forward_steps_remaining
                )
                num_input_embedding_forward_steps_remaining -= 1
            else:
                input_embedding_forward_callback = lambda: None
            input_embedding_backward_callback()
            num_remaining_s_pass -= 1

        if (
            parallel_state.get_pipeline_model_parallel_rank()
            != parallel_state.get_pipeline_model_parallel_world_size() - 2
        ):
            output_tensor_grad = send_forward_recv_backward(
                output_tensor, send_tensor_shapes, config
            )

        if (
            (not is_bsf[parallel_state.get_pipeline_model_parallel_rank()])
            or (offsets[parallel_state.get_pipeline_model_parallel_rank()] + f < 0)
        ):
            input_embedding_forward_callback()
            if get_args().disable_backward_fusion:
                lm_head_reduce_output_store = reduce_lm_head_res_alg1(
                    num_microbatches - num_remaining_s_pass - 1, *lm_head_res
                )
            else:
                lm_head_reduce_output_store = reduce_lm_head_res_alg2(*lm_head_res)

        if (
            parallel_state.get_pipeline_model_parallel_rank()
            == parallel_state.get_pipeline_model_parallel_world_size() - 2
        ):
            output_tensor_grad = send_forward_recv_backward(
                output_tensor, send_tensor_shapes, config
            )

        input_tensors[0].append(input_tensor)
        output_tensors[0].append(output_tensor)
        deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

    # Run cooldown backward passes.
    for i in range(num_microbatches - num_microbatches_remaining):        
        if (
            (is_bsf[parallel_state.get_pipeline_model_parallel_rank()]
             or (offsets[parallel_state.get_pipeline_model_parallel_rank()] - f > -b))
        ):
            if num_microbatches - num_remaining_s_pass >= num_warmup_s_pass[0] + 2 + warmup_offset:
                input_embedding_backward_callback = input_embedding_backward_step_helper(
                    num_microbatches - num_input_embedding_backward_steps_remaining
                )
                num_input_embedding_backward_steps_remaining -= 1
            else:
                input_embedding_backward_callback = lambda: None
            if num_remaining_s_pass >= 1 - warmup_offset:
                receive_lm_head_input_callback = receive_lm_head_input(num_microbatches - num_remaining_s_pass)

        # Enable async grad reduction in the last backward pass
        # Note: If grad sync function is provided, only enable
        # async grad reduction in first pipeline stage. Other
        # pipeline stages do grad reduction during pipeline
        # bubble.
        if i == num_microbatches - num_microbatches_remaining - 1:
            if config.grad_sync_func is None or rank == 0:
                enable_grad_sync()
            
        if (
            (get_args().disable_backward_fusion)
            and (parallel_state.is_pipeline_last_stage())
            and (i + num_microbatches_remaining + 1 < num_microbatches)
        ):
            loss_calculation_helper(i + num_microbatches_remaining + 1)

        input_tensor_grad, post_process = backward_step_helper(
            i + num_microbatches_remaining, output_tensor_grad, run_timer,
        )

        s_executed = False

        if is_bsf[parallel_state.get_pipeline_model_parallel_rank()]:
            if num_remaining_s_pass >= 1 - warmup_offset:
                lm_head_inputs = receive_lm_head_input_callback()
                lm_head_res = lm_head_step_helper(num_microbatches - num_remaining_s_pass, lm_head_inputs, run_timer)
                num_remaining_s_pass -= 1
                s_executed = True
            input_embedding_backward_callback()

        if not parallel_state.is_pipeline_last_stage():
            send_backward(input_tensor_grad, recv_tensor_shapes, config)

        if (
            s_executed
            and (offsets[parallel_state.get_pipeline_model_parallel_rank()] + f >= 0)
        ):
            if get_args().disable_backward_fusion:
                lm_head_reduce_output_store = reduce_lm_head_res_alg1(
                    num_microbatches - num_remaining_s_pass - 1, *lm_head_res
                )
            else:
                lm_head_reduce_output_store = reduce_lm_head_res_alg2(*lm_head_res)
            s_executed = False

        if parallel_state.is_pipeline_last_stage():
            send_backward(input_tensor_grad, recv_tensor_shapes, config)

        if (
            (not is_bsf[parallel_state.get_pipeline_model_parallel_rank()]
             and (offsets[parallel_state.get_pipeline_model_parallel_rank()] - f <= -b))
        ):
            if num_microbatches - num_remaining_s_pass >= num_warmup_s_pass[0] + 2 + warmup_offset:
                input_embedding_backward_callback = input_embedding_backward_step_helper(
                    num_microbatches - num_input_embedding_backward_steps_remaining
                )
                num_input_embedding_backward_steps_remaining -= 1
            else:
                input_embedding_backward_callback = lambda: None
            if num_remaining_s_pass >= 1 - warmup_offset:
                receive_lm_head_input_callback = receive_lm_head_input(num_microbatches - num_remaining_s_pass)

        if not is_bsf[parallel_state.get_pipeline_model_parallel_rank()]:
            if num_remaining_s_pass >= 1 - warmup_offset:
                lm_head_inputs = receive_lm_head_input_callback()
                lm_head_res = lm_head_step_helper(num_microbatches - num_remaining_s_pass, lm_head_inputs, run_timer)
                num_remaining_s_pass -= 1
                s_executed = True
            input_embedding_backward_callback()
        
        if (
            parallel_state.get_pipeline_model_parallel_rank()
            != parallel_state.get_pipeline_model_parallel_world_size() - 2
        ):
            if i + 1 < num_microbatches - num_microbatches_remaining:
                output_tensor_grad = recv_backward(
                    send_tensor_shapes, config
                )

        if s_executed:
            if get_args().disable_backward_fusion:
                lm_head_reduce_output_store = reduce_lm_head_res_alg1(
                    num_microbatches - num_remaining_s_pass - 1, *lm_head_res
                )
            else:
                lm_head_reduce_output_store = reduce_lm_head_res_alg2(*lm_head_res)

        if (
            parallel_state.get_pipeline_model_parallel_rank()
            == parallel_state.get_pipeline_model_parallel_world_size() - 2
        ):
            if i + 1 < num_microbatches - num_microbatches_remaining:
                output_tensor_grad = recv_backward(
                    send_tensor_shapes, config
                )
    
    while num_input_embedding_backward_steps_remaining > 0:
        input_embedding_backward_step_helper(
            num_microbatches - num_input_embedding_backward_steps_remaining
        )()
        num_input_embedding_backward_steps_remaining -= 1

    if not get_args().disable_backward_fusion:
        post_process()

        lm_head_inputs = receive_lm_head_input(num_microbatches)()
        lm_head_step_helper(num_microbatches, lm_head_inputs, run_timer)

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



