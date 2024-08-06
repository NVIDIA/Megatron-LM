import contextlib
import itertools
from typing import Iterator, List, Union

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from megatron import core, get_args, get_num_microbatches, print_rank_0
from megatron.core import parallel_state
from megatron.core.utils import get_model_config, get_model_type
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_next_rank,
    get_pipeline_model_parallel_prev_rank,
)
from megatron.core.pipeline_parallel.schedules import (
    recv_forward,
    send_forward,
    recv_backward,
    send_backward,
    deallocate_output_tensor,
    forward_step,
    backward_step,
    get_tensor_shapes,
)

from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel import p2p_communication

# The BitPipe schedule with direct concatenation
def forward_backward_pipelining_with_BitPipe(
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
):
    assert isinstance(
        model, list
    ), "hybrid pipeline parallelism expected model chunking"
    assert all(
        isinstance(chunk, torch.nn.Module) for chunk in model
    ), "invalid model chunking"
    assert isinstance(
        data_iterator, list
    ), "hybrid pipeline parallelism expected each model chunk to have a data iterator"  # 循环需要

    config = get_model_config(model[0])
    if config.overlap_p2p_comm and config.batch_p2p_comm:
        raise ValueError("Can not use both overlap_p2p_comm and batch_p2p_comm")

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if no_sync_func is None and all(isinstance(chunk, torchDDP) for chunk in model):

        def multi_no_sync():
            stack = contextlib.ExitStack()
            for chunk in model:
                stack.enter_context(chunk.no_sync())
            return stack

        no_sync_func = multi_no_sync
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

    # Model chunk IDs with synchronized grads
    synchronized_model_chunks = set()

    input_tensors = [[] for _ in range(len(model))]
    output_tensors = [[] for _ in range(len(model))]
    forward_data_store = []
    if not forward_only:
        output_tensor_grads = [[] for _ in range(len(model))]

    pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank() 

    if num_microbatches % pipeline_parallel_size != 0:
        msg = f"number of microbatches ({num_microbatches}) is not divisible by "
        msg += f"pipeline-model-parallel-size ({pipeline_parallel_size}) "
        msg += "when using bidirectional schedule"
        raise RuntimeError(msg)

    model_type = get_model_type(model[0])
    if model_type == ModelType.encoder_and_decoder:
        raise RuntimeError(
            "Bidirectional is not supported with an encoder and decoder model."
        )

    if decoder_seq_length is not None and decoder_seq_length != tensor_shape[0]:
        raise RuntimeError(
            "Bidirectional is not supported with a different decoder sequence length."
        )

    tensor_shape = (seq_length, micro_batch_size, config.hidden_size)
    if config.sequence_parallel:
        tensor_shape[0] = (
            tensor_shape[0] // parallel_state.get_tensor_model_parallel_world_size()
        )

    # Compute number of warmup and remaining microbatches.
    num_model_chunks = len(model) # 4,if BitPipe
    total_num_microbatches = num_microbatches * (num_model_chunks//2)
    # use 2*pipeline_parallel_size as a basic unit, loop if n_loop>0
    n_loop =total_num_microbatches//pipeline_parallel_size//2 -1

    if forward_only:
        num_warmup_microbatches = total_num_microbatches
    else:
        # Run all forward passes and then all backward passes if number of
        # microbatches is just the number of pipeline stages.
        # Otherwise, perform (num_model_chunks-1)*pipeline_parallel_size on
        # all workers, followed by more microbatches after depending on
        # stage ID (more forward passes for earlier stages, later stages can
        # immediately start with 2F1B).
        if total_num_microbatches == pipeline_parallel_size:
            num_warmup_microbatches = total_num_microbatches
        else:
            num_warmup_microbatches = pipeline_parallel_size +pipeline_parallel_size//2

        num_warmup_microbatches += (
            pipeline_parallel_rank
            if pipeline_parallel_rank < pipeline_parallel_size // 2
            else pipeline_parallel_size - 1 - pipeline_parallel_rank
        )
    unit_remaining = 2*pipeline_parallel_size - num_warmup_microbatches
    num_microbatches_mid=n_loop*pipeline_parallel_size*2
    num_microbatches_remaining = total_num_microbatches - num_warmup_microbatches-num_microbatches_mid

    assert config.num_microbatches_with_partial_activation_checkpoints is None

    # Synchronize params for first two model chunks
    if config.param_sync_func is not None:
        config.param_sync_func(model[0].parameters())
        config.param_sync_func(model[1].parameters())

    def get_microbatch(total_num_microbatches):
        microbatch_id01 = [[] for _ in range(num_model_chunks)]
        for i in range(total_num_microbatches):
            model_chunk_id = get_model_chunk_id(i)
            microbatch_id01[model_chunk_id].append(i)
        return microbatch_id01

    def get_microbatch_idx(total_num_microbatches, pipeline_parallel_rank):
        # num_microbatches=8 
        # 0,    1, 2, 6, 3, 7, 4,    5
        #    0, 2, 1, 3, 6, 4, 7, 5
        #    2, 0, 3, 1, 4, 6, 5, 7
        # 2,    3, 0, 4, 1, 5, 6,    7
        microbatch_idx = []
        microbatch_id01 = get_microbatch(total_num_microbatches)
        i_loop =total_num_microbatches//pipeline_parallel_size//2
        # chunk_size=total_num_microbatches//num_model_chunks
        num_unit = pipeline_parallel_size // 2
        i_half = pipeline_parallel_rank // num_unit
        num_initial = (
            num_unit - pipeline_parallel_rank
            if pipeline_parallel_rank < num_unit
            else pipeline_parallel_rank - num_unit+1
        )
        # [0, 1, 2, 10, 3, 11, 8, 9, 4, 5, 6, 14, 7, 15, 12, 13]
        # [0, 2, 1, 3, 10, 8, 11, 9, 4, 6, 5, 7, 14, 12, 15, 13]
        # [2, 0, 3, 1, 8, 10, 9, 11, 6, 4, 7, 5, 12, 14, 13, 15]
        # [2, 3, 0, 8, 1, 9, 10, 11, 6, 7, 4, 12, 5, 13, 14, 15]

        for j in range(i_loop):
            for i in range(num_initial):
                microbatch_idx.append(microbatch_id01[i_half][j*num_unit+i])
            for i in range(num_unit- num_initial): 
                microbatch_idx.append(microbatch_id01[1-i_half][j*num_unit+i])
                microbatch_idx.append(microbatch_id01[i_half][j*num_unit+i+num_initial])
            for i in range(num_initial):
                microbatch_idx.append(microbatch_id01[1-i_half][j*num_unit+i+(num_unit- num_initial)])
                microbatch_idx.append(microbatch_id01[3-i_half][j*num_unit+i])
            for i in range(num_unit- num_initial): 
                microbatch_idx.append(microbatch_id01[2+i_half][j*num_unit+i])  
                microbatch_idx.append(microbatch_id01[3-i_half][j*num_unit+i+num_initial])                
            for i in range(num_initial):
                microbatch_idx.append(microbatch_id01[2+i_half][j*num_unit+i+(num_unit- num_initial)])

        return microbatch_idx


    def get_bkmicrobatch_idx(total_num_microbatches, pipeline_parallel_rank):
        microbatch_idx=[]
        microbatch_id01 = get_microbatch(total_num_microbatches)
        i_loop =total_num_microbatches//pipeline_parallel_size//2

        num_unit = pipeline_parallel_size // 2
        i_half = pipeline_parallel_rank // num_unit
        num_initial = (
            num_unit - pipeline_parallel_rank
            if pipeline_parallel_rank < num_unit
            else pipeline_parallel_rank - num_unit+1
        )
        # [8, 9, 10, 2, 11, 3, 0, 1, 12, 13, 14, 6, 15, 7, 4, 5]
        # [8, 10, 9, 11, 2, 0, 3, 1, 12, 14, 13, 15, 6, 4, 7, 5]
        # [10, 8, 11, 9, 0, 2, 1, 3, 14, 12, 15, 13, 4, 6, 5, 7]
        # [10, 11, 8, 0, 9, 1, 2, 3, 14, 15, 12, 4, 13, 5, 6, 7]

        for j in range(i_loop):
            for k in range(num_initial): 
                microbatch_idx.append(microbatch_id01[2+i_half][j*num_unit+k])
            for k in range(num_unit-num_initial): 
                microbatch_idx.append(microbatch_id01[3-i_half][j*num_unit+k])
                microbatch_idx.append(microbatch_id01[2+i_half][j*num_unit+k+num_initial])
            for k in range(num_initial):
                microbatch_idx.append(microbatch_id01[3-i_half][(j*num_unit+k+num_unit- num_initial)])
                microbatch_idx.append(microbatch_id01[1-i_half][j*num_unit+k])
            for k in range(num_unit-num_initial): 
                microbatch_idx.append(microbatch_id01[i_half][j*num_unit+k]) 
                microbatch_idx.append(microbatch_id01[1-i_half][j*num_unit+k+num_initial])
            for k in range(num_initial):
                microbatch_idx.append(microbatch_id01[i_half][j*num_unit+k+num_unit- num_initial])
        # Eager sync
        if pipeline_parallel_rank==pipeline_parallel_size // 2 or pipeline_parallel_rank==pipeline_parallel_size // 2 -1:
            microbatch_idx.append(-1) 
        else: # second last sync
            microbatch_idx.insert(-1,-1)
        microbatch_idx.append(-1) # last sync
        return microbatch_idx
    
    def get_model_chunk_id(microbatch_id):  # 0,1,4,5  #2,3,6,7 # 8,9,12,13  #10,11,14,15
        """Helper method to get the model chunk ID given the iteration number."""
        microbatch_id_in_group = microbatch_id % (pipeline_parallel_size)
        chunk_offset =0 if microbatch_id<(total_num_microbatches//2) else 2
        model_chunk_id = microbatch_id_in_group // (pipeline_parallel_size // 2)
        model_chunk_id += chunk_offset
        if microbatch_id ==-1:
            model_chunk_id =-1
        return model_chunk_id

    def is_first_microbatch_for_model_chunk(microbatch_id: int) -> bool:
        """Check if an iteration is the first for a model chunk."""
        microbatch_id01 = get_microbatch(total_num_microbatches)
        if (
            microbatch_id == microbatch_id01[0][0]
            or microbatch_id == microbatch_id01[1][0]
        ):
            return True
        else:
            return False

    def is_last_microbatch_for_model_chunk(microbatch_id: int) -> bool:
        """Check if an iteration is the last for a model chunk."""
        microbatch_id01 = get_microbatch(total_num_microbatches)

        if (
            microbatch_id == microbatch_id01[0][-1]
            or microbatch_id == microbatch_id01[1][-1]
        ):
            return True
        else:
            return False

    def forward_step_helper(microbatch_id, checkpoint_activations_microbatch,offset):
        """Helper method to run forward step with model split into chunks
        (run set_bidirectional_pipeline_model_parallel_rank() before calling
        forward_step())."""

        model_chunk_id = get_model_chunk_id(microbatch_id)
        # forward step
        if parallel_state.is_pipeline_first_stage():
            if len(input_tensors[model_chunk_id]) == len(
                output_tensors[model_chunk_id]
            ):
                input_tensors[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id][-1-offset]
        output_tensor = forward_step(
            forward_step_func,
            data_iterator[model_chunk_id],
            model[model_chunk_id],
            num_microbatches//2,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
        )
        output_tensors[model_chunk_id].append(output_tensor)
        return output_tensor

    def allreduce_gradients(model):
        # Pack the buckets.
        if (
            parallel_state.is_rank_in_bd_group()
            and parallel_state.get_pipeline_model_parallel_world_size() > 1
        ):
            torch.distributed.barrier(group=parallel_state.get_bd_parallel_group())
            buckets = {}
            for param in model.module.parameters():
                if param.requires_grad and param.main_grad is not None:
                    tp = param.data.type()
                    if tp not in buckets:
                        buckets[tp] = []
                    buckets[tp].append(param)

            # For each bucket, all-reduce and copy all-reduced grads.
            for tp in buckets:
                bucket = buckets[tp]
                grads = [param.main_grad.data for param in bucket]
                coalesced = _flatten_dense_tensors(grads)
                coalesced /= torch.distributed.get_world_size(group=parallel_state.get_bd_parallel_group())
                torch.distributed.all_reduce(
                    coalesced, group=parallel_state.get_bd_parallel_group())
                for buf, synced in zip(grads, _unflatten_dense_tensors(
                        coalesced, grads)):
                    buf.copy_(synced)


    def backward_step_helper(microbatch_id):
        """Helper method to run backward step with model split into chunks
        (run set_bidirectional_pipeline_model_parallel_rank() before calling
        backward_step())."""
        # launch grad synchronization (default)
        model_chunk_id = get_model_chunk_id(microbatch_id)
        if config.grad_sync_func is None and is_last_microbatch_for_model_chunk(
            microbatch_id
        ):
            enable_grad_sync()
            synchronized_model_chunks.add(model_chunk_id)
            
        if parallel_state.is_pipeline_last_stage():
            if len(output_tensor_grads[model_chunk_id]) == 0:
                output_tensor_grads[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id].pop(0)
        output_tensor = output_tensors[model_chunk_id].pop(0)
        output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)
        input_tensor_grad = backward_step(
            input_tensor, output_tensor, output_tensor_grad, model_type, config
        )

        # launch grad synchronization (custom grad sync)
        # Note: Asynchronous communication tends to slow down compute.
        # To reduce idling from mismatched microbatch times, we launch
        # asynchronous communication at the same time across the
        # pipeline-parallel group.
        if config.grad_sync_func is not None:
            grad_sync_microbatch_id = microbatch_id
            if grad_sync_microbatch_id >= 0 and is_last_microbatch_for_model_chunk(
                grad_sync_microbatch_id
            ):
                grad_sync_chunk_id = get_model_chunk_id(grad_sync_microbatch_id)
                enable_grad_sync()
                config.grad_sync_func(model[grad_sync_chunk_id].parameters())
                synchronized_model_chunks.add(grad_sync_chunk_id)

        disable_grad_sync()      
        return input_tensor_grad

    # Run warmup forward passes.
    if pipeline_parallel_rank < pipeline_parallel_size // 2:
        parallel_state.set_virtual_pipeline_model_parallel_rank(0)
        input_tensors[0].append(p2p_communication.recv_forward(tensor_shape, config))
    else:
        parallel_state.set_virtual_pipeline_model_parallel_rank(1)
        input_tensors[1].append(p2p_communication.recv_forward(tensor_shape, config))

    microbatch_idx = get_microbatch_idx(total_num_microbatches, pipeline_parallel_rank)
    microbatch_idx_b = get_bkmicrobatch_idx(total_num_microbatches, pipeline_parallel_rank) 

   # the initial warmup
    for k in range(num_warmup_microbatches):
        forward_model_chunk_id = get_model_chunk_id(microbatch_idx[k])
        parallel_state.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)
        output_tensor = forward_step_helper(microbatch_idx[k], None,0)  

        if k < total_num_microbatches - 1:  
            next_forward_model_chunk_id = get_model_chunk_id(
                microbatch_idx[k + 1]
            )
        else:
            next_forward_model_chunk_id = 1

        recv_prev = True
        recv_next = True
        if parallel_state.is_pipeline_first_stage():
            recv_prev = False

        # Don't send tensor downstream if on last stage.
        if parallel_state.is_pipeline_last_stage():
            output_tensor = None

        v_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
        v_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()
        # Send and receive tensors as appropriate (send tensors computed
        # in this iteration; receive tensors for next iteration).
        if not config.overlap_p2p_comm:
            if forward_model_chunk_id == next_forward_model_chunk_id:
                if parallel_state.is_pipeline_first_stage():
                    recv_prev = False
                if k==total_num_microbatches-1 and forward_only: # for evaluate
                    recv_prev = False
                input_tensor = p2p_communication.send_forward_recv_forward(
                    output_tensor,
                    recv_prev=recv_prev,
                    tensor_shape=tensor_shape,
                    config=config,
                )
            else:
                if (parallel_state.is_pipeline_last_stage(ignore_virtual=True) and next_forward_model_chunk_id == v_size-2) or (parallel_state.is_pipeline_first_stage(ignore_virtual=True) and next_forward_model_chunk_id == v_size-1):
                    detached_output_tensor = output_tensor.detach()
                    detached_output_tensor.requires_grad_()
                    input_tensor=detached_output_tensor # last stage of chunk0 and first stage of chunk2 are in the same device 
                else:
                    if parallel_state.is_pipeline_last_stage():
                        recv_next = False
                    if k==total_num_microbatches-1 and forward_only: # for evaluate
                        recv_next = False

                    input_tensor = p2p_communication.send_forward_recv_forward_bd0(
                        output_tensor,
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        config=config,
                    )
            input_tensors[next_forward_model_chunk_id].append(input_tensor)

        deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

    # the mid_unit to loop, if exists
    if not forward_only:
        for j in range(n_loop): 
            offset=j*pipeline_parallel_size*2
            #1 Run 1F, 1B, respectively
            for k in range(unit_remaining*2):
                forward_k = k//2 + num_warmup_microbatches+offset
                backward_k = k//2+offset
                if k%2==0:
                # Forward pass.
                    forward_model_chunk_id = get_model_chunk_id(microbatch_idx[forward_k])
                    parallel_state.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)
                    output_tensor = forward_step_helper(microbatch_idx[forward_k], None,0)
                    if parallel_state.is_pipeline_last_stage():
                        output_tensor = None
                    recv_next = True
                    next_backward_model_chunk_id = get_model_chunk_id(microbatch_idx_b[backward_k])
                    # Communicate tensors.
                    if not parallel_state.is_pipeline_last_stage():# recv grad if not last stage
                        output_tensor_grad= p2p_communication.send_forward_recv_backward(
                        output_tensor,
                        tensor_shape=tensor_shape,
                        config=config,
                        )
                        output_tensor_grads[next_backward_model_chunk_id].append(
                            output_tensor_grad
                        )
                    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
                else:
                    # Backward pass.
                    backward_model_chunk_id = get_model_chunk_id(microbatch_idx_b[backward_k])
                    parallel_state.set_virtual_pipeline_model_parallel_rank(
                        backward_model_chunk_id
                    )
                    input_tensor_grad = backward_step_helper(microbatch_idx_b[backward_k])

                    if parallel_state.is_pipeline_first_stage():
                        input_tensor_grad = None

                    # Determine if peers are sending, and where in data structure to put
                    # received tensors.
                    recv_prev = True
                    next_forward_model_chunk_id = get_model_chunk_id(microbatch_idx[forward_k + 1])
                    next_backward_model_chunk_id = get_model_chunk_id(microbatch_idx_b[backward_k + 1])
                    # If last iteration, don't receive; we already received one extra
                    # before the start of the for loop.
                    # Communicate tensors.
                    if k == unit_remaining*2 - 1:
                        output_tensor_grad = p2p_communication.send_backward_recv_backward_bd(
                            input_tensor_grad,
                            recv_prev=recv_prev,
                            tensor_shape=tensor_shape,
                            config=config,
                        )
                        if recv_prev:
                            output_tensor_grads[next_backward_model_chunk_id].append(
                                output_tensor_grad
                            )
                    else:
                        input_tensor = p2p_communication.send_backward_recv_forward(
                            input_tensor_grad,
                            tensor_shape=tensor_shape,
                            config=config,
                        )
                        input_tensors[next_forward_model_chunk_id].append(input_tensor)

            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

            #2 Run cooldown backward passes (num_warmup_microbatches-unit_remaining+1).
            for k in range(num_warmup_microbatches-unit_remaining+1):  
                backward_k = k+offset+unit_remaining   
                backward_model_chunk_id = get_model_chunk_id(microbatch_idx_b[backward_k])
                parallel_state.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)
                input_tensor_grad = backward_step_helper(microbatch_idx_b[backward_k])
  
                recv_next = True
                recv_prev = True
                if parallel_state.is_pipeline_first_stage():
                    recv_prev = False
                    input_tensor_grad = None

                if k == (num_warmup_microbatches-unit_remaining):
                    next_forward_model_chunk_id = get_model_chunk_id(microbatch_idx[offset+2*pipeline_parallel_size])
                    if not parallel_state.is_pipeline_first_stage():# recv input is not fisrt stage
                        input_tensor=p2p_communication.send_backward_recv_forward(input_tensor_grad,tensor_shape=tensor_shape,
                            config=config,)
                        input_tensors[next_forward_model_chunk_id].append(input_tensor)
                else:
                    next_backward_model_chunk_id = get_model_chunk_id(microbatch_idx_b[backward_k+1])
                    if (parallel_state.is_pipeline_last_stage(ignore_virtual=True) and next_backward_model_chunk_id == 0) or (parallel_state.is_pipeline_first_stage(ignore_virtual=True) and next_backward_model_chunk_id == 1):
                        output_tensor_grads[next_backward_model_chunk_id].append(input_tensor_grad)
                    else:
                        output_tensor_grads[next_backward_model_chunk_id].append(
                            p2p_communication.send_backward_recv_backward_bd(
                                input_tensor_grad,
                                recv_prev=recv_prev,
                                tensor_shape=tensor_shape,
                                config=config,
                            )
                        )

            #3 1F,1B of cooldown
            for k in range(2*(unit_remaining-1)):
                # Forward pass.
                forward_k = k//2 + 2*pipeline_parallel_size+offset
                backward_k = k//2 +num_warmup_microbatches+1+offset
                if k%2==0:
                    # Forward pass.
                    forward_model_chunk_id = get_model_chunk_id(microbatch_idx[forward_k])
                    parallel_state.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)
                    output_tensor = forward_step_helper(microbatch_idx[forward_k], None,0)
                    if parallel_state.is_pipeline_last_stage():
                        output_tensor = None
                    recv_next = True
                    next_backward_model_chunk_id = get_model_chunk_id(microbatch_idx_b[backward_k])
                    # Communicate tensors.
                    if not parallel_state.is_pipeline_last_stage():# recv grad if not last stage
                        output_tensor_grad= p2p_communication.send_forward_recv_backward(
                        output_tensor,
                        tensor_shape=tensor_shape,
                        config=config,
                        )
                        output_tensor_grads[next_backward_model_chunk_id].append(
                            output_tensor_grad
                        )
                    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
                else:
                    # Backward pass.
                    backward_model_chunk_id = get_model_chunk_id(microbatch_idx_b[backward_k])
                    parallel_state.set_virtual_pipeline_model_parallel_rank(
                        backward_model_chunk_id
                    )
                    input_tensor_grad = backward_step_helper(microbatch_idx_b[backward_k])
                    recv_prev = True
                    next_forward_model_chunk_id = get_model_chunk_id(microbatch_idx[forward_k + 1])
                    if parallel_state.is_pipeline_first_stage():
                        input_tensor_grad = None
                        recv_prev =False
                    # If last iteration, don't receive; we already received one extra
                    # before the start of the for loop.
                    # Communicate tensors.
                    if not parallel_state.is_pipeline_first_stage():
                        input_tensor = p2p_communication.send_backward_recv_forward(
                            input_tensor_grad,
                            tensor_shape=tensor_shape,
                            config=config,
                        )
                        input_tensors[next_forward_model_chunk_id].append(input_tensor)
                    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

            #4 warmup again
            for k in range(num_warmup_microbatches-unit_remaining+1):
                forward_k = k + 2*pipeline_parallel_size+offset+unit_remaining-1
                forward_model_chunk_id = get_model_chunk_id(microbatch_idx[forward_k])
                parallel_state.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)
                output_tensor = forward_step_helper(microbatch_idx[forward_k], None,0)  

                # Determine if tensor should be received from previous stage.
                if forward_k < total_num_microbatches - 1:  # for evaluate
                    next_forward_model_chunk_id = get_model_chunk_id(
                        microbatch_idx[forward_k + 1]
                    )
                else:
                    next_forward_model_chunk_id = 1

                recv_prev = True
                recv_next = True
                if parallel_state.is_pipeline_first_stage():
                    recv_prev = False

                # Don't send tensor downstream if on last stage.
                if parallel_state.is_pipeline_last_stage():
                    output_tensor = None

                v_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
                v_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()
                # Send and receive tensors as appropriate (send tensors computed
                # in this iteration; receive tensors for next iteration).
                if not config.overlap_p2p_comm:
                    if (parallel_state.is_pipeline_last_stage(ignore_virtual=True) and next_forward_model_chunk_id == v_size-2) or (parallel_state.is_pipeline_first_stage(ignore_virtual=True) and next_forward_model_chunk_id == v_size-1):
                        detached_output_tensor = output_tensor.detach()
                        detached_output_tensor.requires_grad_()
                        input_tensor=detached_output_tensor 
                    else:
                        if parallel_state.is_pipeline_last_stage():
                            recv_next = False
                        if forward_k==total_num_microbatches-1 and forward_only: 
                            recv_next = False

                        input_tensor = p2p_communication.send_forward_recv_forward_bd0(
                            output_tensor,
                            recv_next=recv_next,
                            tensor_shape=tensor_shape,
                            config=config,
                        )
                    input_tensors[next_forward_model_chunk_id].append(input_tensor)

                deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
                
    if not forward_only:
        # Run 1F, 1B, respectively, last.
        for k in range(2*unit_remaining):
            # Forward pass.
            forward_k = k//2 + num_warmup_microbatches+num_microbatches_mid
            backward_k = k//2 +num_microbatches_mid
            if k%2==0:
                # Forward pass.
                forward_model_chunk_id = get_model_chunk_id(microbatch_idx[forward_k])
                parallel_state.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)
                output_tensor = forward_step_helper(microbatch_idx[forward_k], None,0)
                if parallel_state.is_pipeline_last_stage():
                    output_tensor = None
                recv_next = True
                next_backward_model_chunk_id = get_model_chunk_id(microbatch_idx_b[backward_k])
                # Communicate tensors.
                if not parallel_state.is_pipeline_last_stage():
                    output_tensor_grad= p2p_communication.send_forward_recv_backward(
                    output_tensor,
                    tensor_shape=tensor_shape,
                    config=config,
                    )
                    output_tensor_grads[next_backward_model_chunk_id].append(
                        output_tensor_grad
                    )
                deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
            else:
                # Backward pass.
                backward_model_chunk_id = get_model_chunk_id(microbatch_idx_b[backward_k])
                parallel_state.set_virtual_pipeline_model_parallel_rank(
                    backward_model_chunk_id
                )
                input_tensor_grad = backward_step_helper(microbatch_idx_b[backward_k])

                if parallel_state.is_pipeline_first_stage():
                    input_tensor_grad = None

                # Determine if peers are sending, and where in data structure to put
                # received tensors.
                recv_prev = True
                next_backward_model_chunk_id = get_model_chunk_id(microbatch_idx_b[backward_k + 1])
                if k < unit_remaining*2 - 1:  
                    next_forward_model_chunk_id = get_model_chunk_id(
                        microbatch_idx[forward_k + 1]
                    )
                else:
                    next_forward_model_chunk_id = 0

                # Communicate tensors.
                if k == unit_remaining*2 - 1:
                    output_tensor_grad = p2p_communication.send_backward_recv_backward_bd(
                        input_tensor_grad,
                        recv_prev=recv_prev,
                        tensor_shape=tensor_shape,
                        config=config,
                    )
                    if recv_prev:
                        output_tensor_grads[next_backward_model_chunk_id].append(
                            output_tensor_grad
                        )
                else:
                    input_tensor = p2p_communication.send_backward_recv_forward(
                        input_tensor_grad,
                        tensor_shape=tensor_shape,
                        config=config,
                    )
                    input_tensors[next_forward_model_chunk_id].append(input_tensor)

        deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

    # Run cooldown backward passes (flush out pipeline).
    if not forward_only:
        for k in range(2*pipeline_parallel_size-unit_remaining+2):  
            backward_k = k+num_microbatches_mid+unit_remaining
            backward_model_chunk_id = get_model_chunk_id(microbatch_idx_b[backward_k])
            if backward_k < total_num_microbatches+1:  
                next_backward_model_chunk_id = get_model_chunk_id(microbatch_idx_b[backward_k+ 1])
            else:
                next_backward_model_chunk_id = -1
            if backward_model_chunk_id ==-1: # eager sync
                offset =range(num_model_chunks//2) if pipeline_parallel_rank<pipeline_parallel_size//2 else reversed(range(num_model_chunks//2))
                if backward_k<total_num_microbatches+1:
                    for i_chunk in offset:
                        allreduce_gradients(model[num_model_chunks//2+i_chunk])
                    if not next_backward_model_chunk_id ==-1:
                        output_tensor_grads[next_backward_model_chunk_id].append(p2p_communication.recv_backward(tensor_shape=tensor_shape,
                                config=config))
                elif backward_k==total_num_microbatches+1:
                    for i_chunk in offset:
                        allreduce_gradients(model[i_chunk])
            else:
                input_tensor_grad = backward_step_helper(microbatch_idx_b[backward_k])

                parallel_state.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)
                recv_next = True
                recv_prev = True
                if parallel_state.is_pipeline_first_stage():
                    recv_prev = False
                    input_tensor_grad = None

                if backward_k == (total_num_microbatches - 2):
                    if next_backward_model_chunk_id ==-1:
                        recv_next = False
                        output_tensor_grad = p2p_communication.send_backward_recv_backward(
                                input_tensor_grad,
                                recv_next=recv_next,
                                tensor_shape=tensor_shape,
                                config=config,
                            )
                    else:
                        output_tensor_grads[next_backward_model_chunk_id].append(
                            p2p_communication.send_backward_recv_backward_bd(
                                input_tensor_grad,
                                recv_prev=recv_prev,
                                tensor_shape=tensor_shape,
                                config=config,
                            )
                        )
                elif backward_k == (total_num_microbatches - 1):
                    recv_next = False
                    output_tensor_grad = p2p_communication.send_backward_recv_backward(
                            input_tensor_grad,
                            recv_next=recv_next,
                            tensor_shape=tensor_shape,
                            config=config,
                        )
                elif backward_k == (total_num_microbatches):
                    recv_next = False
                    output_tensor_grad = p2p_communication.send_backward_recv_backward(
                            input_tensor_grad,
                            recv_next=recv_next,
                            tensor_shape=tensor_shape,
                            config=config,
                        )
                else:
                    if backward_model_chunk_id == next_backward_model_chunk_id:
                        output_tensor_grads[next_backward_model_chunk_id].append(
                            p2p_communication.send_backward_recv_backward(
                                input_tensor_grad,
                                recv_next=recv_next,
                                tensor_shape=tensor_shape,
                                config=config,
                            )
                        )
                    else:
                        if (parallel_state.is_pipeline_last_stage(ignore_virtual=True) and next_backward_model_chunk_id == 0) or (parallel_state.is_pipeline_first_stage(ignore_virtual=True) and next_backward_model_chunk_id == 1):
                            output_tensor_grads[next_backward_model_chunk_id].append(input_tensor_grad)
                        else:
                            output_tensor_grads[next_backward_model_chunk_id].append(
                                p2p_communication.send_backward_recv_backward_bd(
                                    input_tensor_grad,
                                    recv_prev=recv_prev,
                                    tensor_shape=tensor_shape,
                                    config=config,
                                )
                            )

    # Launch any remaining grad reductions
    enable_grad_sync()
    if config.grad_sync_func is not None:
        params = []
        for model_chunk_id in range(num_model_chunks):
            if model_chunk_id not in synchronized_model_chunks:
                params.extend(model[model_chunk_id].parameters())
                synchronized_model_chunks.add(model_chunk_id)
        if params:
            config.grad_sync_func(params)

    return forward_data_store