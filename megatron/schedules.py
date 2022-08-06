# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

from contextlib import contextmanager
import torch
from torch.autograd.variable import Variable
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron import get_args
from megatron import get_num_microbatches
from megatron import get_timers
from megatron import p2p_communication
from megatron.core import mpu
from megatron.utils import unwrap_model
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import Float16Module
from megatron.model import ModelType


def get_forward_backward_func():
    args = get_args()
    if mpu.get_pipeline_model_parallel_world_size() > 1:
        if args.virtual_pipeline_model_parallel_size is not None:
            forward_backward_func = forward_backward_pipelining_with_interleaving
            assert get_num_microbatches() % \
                args.pipeline_model_parallel_size == 0, \
                'number of microbatches (%d) is not divisible by pipeline-' \
                'model-parallel-size (%d) when using interleaved schedule' % (
                    get_num_microbatches(),
                    args.pipeline_model_parallel_size,
                )
        else:
            forward_backward_func = forward_backward_pipelining_without_interleaving
    else:
        forward_backward_func = forward_backward_no_pipelining
    return forward_backward_func

def deallocate_output_tensor(out):
    '''Pseudo-deallocate (i.e., set to scalar) the output tensor's '.data' field.

    This method should be called right after the output tensor has been
    sent to the next pipeline stage. At this point, the output tensor is
    only useful for its '.grad_fn' field, and not its '.data'.
    '''
    if out is None:
        return
    # If out is a list of tensors, only deallocate the first tensor in the list.
    if isinstance(out, list):
        out = out[0]
    assert isinstance(out, torch.Tensor), \
        "expected Tensor, found %s." % type(out).__name__
    assert out._base is None, \
        "counter-productive to free a view of another tensor."
    out.data = torch.empty(
        (1,),
        device = out.device,
        dtype = out.dtype,
    )
        
def custom_backward(output, grad_output, lbl_loss=None):
    '''Directly call C++ autograd engine.

    To make the 'deallocate_output_tensor' (above) optimization work, the C++
    autograd engine must be called directly, bypassing Pytorch's
    torch.autograd.backward. Pytorch's 'backward' checks that the output and
    grad have the same shape, while C++'s 'backward' does not.
    '''

    assert output.numel() == 1, \
        "output should be pseudo-'freed' in schedule, to optimize memory"
    assert isinstance(output, torch.Tensor), \
        "output == '%s'." % type(output).__name__
    assert isinstance(grad_output, (torch.Tensor, type(None))), \
        "grad_output == '%s'." % type(grad_output).__name__

    # Handle scalar output
    if grad_output is None:
        assert output.numel() == 1, "implicit grad requires scalar output."
        grad_output = torch.ones_like(
            output,
            memory_format = torch.preserve_format,
        )
    grad_lbl_loss = None
    if lbl_loss is not None:
        assert lbl_loss.numel() == 1, "implicit grad requires scalar output."
        grad_lbl_loss = torch.ones_like(
            lbl_loss,
            memory_format = torch.preserve_format,
        )
        tensors = (output, lbl_loss)
        grad_tensors = (grad_output, grad_lbl_loss)
    else:
        tensors = (output,)
        grad_tensors = (grad_output,)

    # Call c++ engine [ see torch/csrc/autograd/python_engine.cpp ]
    Variable._execution_engine.run_backward(
        tensors = tensors,
        grad_tensors = grad_tensors,
        keep_graph = False,
        create_graph = False,
        inputs = tuple(),
        allow_unreachable=True,
        accumulate_grad=True,
    )
        

def forward_step(forward_step_func,
                 data_iterator,
                 model,
                 input_tensor,
                 forward_data_store,
                 timers,
                 collect_non_loss_data=False):
    """Forward step for passed-in model.

    If first stage, input tensor is obtained from data_iterator, otherwise
    passed-in input_tensor is used.

    Returns output tensor."""
    args = get_args()

    if timers is not None:
        timers('forward-compute', log_level=2).start()
    unwrapped_model = unwrap_model(
        model, (torchDDP, LocalDDP, Float16Module))

    unwrap_output_tensor = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_output_tensor = True

    unwrapped_model.set_input_tensor(input_tensor)
    output_tensor, loss_func = forward_step_func(data_iterator, model)
    lbl_loss = None
    if mpu.is_pipeline_last_stage():
        if not collect_non_loss_data:
            output_tensor = loss_func(output_tensor)
            loss, loss_reduced = output_tensor
            output_tensor = loss / get_num_microbatches()
            forward_data_store.append(loss_reduced)
        else:
            data = loss_func(output_tensor, non_loss_data=True)
            forward_data_store.append(data)
    elif args.model_type == ModelType.encoder_or_decoder_with_lbl:
        lbl_loss, lbl_loss_reduced = loss_func()
        lbl_loss = lbl_loss / get_num_microbatches()
        forward_data_store.append(lbl_loss_reduced)

    if timers is not None:
        timers('forward-compute').stop()

    # If T5 model (or other model with encoder and decoder)
    # and in decoder stack, then send encoder_hidden_state
    # downstream as well.
    if mpu.is_pipeline_stage_after_split() and \
            args.model_type == ModelType.encoder_and_decoder:
        return [output_tensor, input_tensor[-1]]
    if lbl_loss is not None:
        return [output_tensor, lbl_loss]
    if unwrap_output_tensor:
        return output_tensor
    return [output_tensor]


def backward_step(optimizer, input_tensor, output_tensor,
                  output_tensor_grad, timers):
    """Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first
    stage)."""

    # NOTE: This code currently can handle at most one skip connection. It
    # needs to be modified slightly to support arbitrary numbers of skip
    # connections.
    args = get_args()

    if timers is not None:
        timers('backward-compute', log_level=2).start()

    # Retain the grad on the input_tensor.
    unwrap_input_tensor_grad = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_input_tensor_grad = True
    for x in input_tensor:
        if x is not None:
            x.retain_grad()

    if not isinstance(output_tensor, list):
        output_tensor = [output_tensor]
    if not isinstance(output_tensor_grad, list):
        output_tensor_grad = [output_tensor_grad]

    # Backward pass.
    if output_tensor_grad[0] is None:
        output_tensor = optimizer.scale_loss(output_tensor[0])
    if mpu.is_pipeline_last_stage():
        custom_backward(output_tensor[0], output_tensor_grad[0])
    elif args.model_type == ModelType.encoder_or_decoder_with_lbl:
        output_tensor[1] = optimizer.scale_loss(output_tensor[1])
        custom_backward(output_tensor[0], output_tensor_grad[0], lbl_loss=output_tensor[1])
    else:
        custom_backward(output_tensor[0], output_tensor_grad[0])

    # Collect the grad of the input_tensor.
    input_tensor_grad = [None]
    if input_tensor is not None:
        input_tensor_grad = []
        for x in input_tensor:
            if x is None:
                input_tensor_grad.append(None)
            else:
                input_tensor_grad.append(x.grad)

    # Handle single skip connection if it exists (encoder_hidden_state in
    # model with encoder and decoder).
    if mpu.get_pipeline_model_parallel_world_size() > 1 and \
            mpu.is_pipeline_stage_after_split() and \
            args.model_type == ModelType.encoder_and_decoder:
        if output_tensor_grad[1] is not None:
            input_tensor_grad[-1].add_(output_tensor_grad[1])
    if unwrap_input_tensor_grad:
        input_tensor_grad = input_tensor_grad[0]

    if timers is not None:
        timers('backward-compute').stop()

    return input_tensor_grad


@contextmanager
def dummy_handler():
    try:
        yield
    finally:
        pass


def forward_backward_no_pipelining(forward_step_func,
                                   data_iterator, model,
                                   optimizer,
                                   timers,
                                   forward_only,
                                   collect_non_loss_data=False):
    """Run forward and backward passes with no pipeline parallelism
    (no inter-stage communication).

    Returns dictionary with losses."""
    assert len(model) == 1
    model = model[0]

    context_handler = dummy_handler
    if isinstance(model, torchDDP):
        context_handler = model.no_sync

    forward_data_store = []
    input_tensor, output_tensor_grad = None, None
    with context_handler():
        for i in range(get_num_microbatches() - 1):
            output_tensor = forward_step(forward_step_func, data_iterator,
                                         model, input_tensor, forward_data_store,
                                         timers, collect_non_loss_data)
            if not forward_only:
                backward_step(optimizer, input_tensor, output_tensor,
                              output_tensor_grad, timers)

    # Run computation for last microbatch out of context handler (want to
    # synchronize gradients).
    output_tensor = forward_step(forward_step_func, data_iterator,
                                 model, input_tensor, forward_data_store,
                                 timers, collect_non_loss_data)
    if not forward_only:
        backward_step(optimizer, input_tensor, output_tensor,
                      output_tensor_grad, timers)

    return forward_data_store


def forward_backward_pipelining_with_interleaving(forward_step_func,
                                                  data_iterator, model,
                                                  optimizer,
                                                  timers,
                                                  forward_only, 
                                                  collect_non_loss_data=False):
    """Run interleaved 1F1B schedule (model split into model chunks), with
    communication between pipeline stages as needed.

    Returns dictionary with losses if the last stage, empty dict otherwise."""

    args = get_args()

    input_tensors = [[] for _ in range(len(model))]
    output_tensors = [[] for _ in range(len(model))]
    forward_data_store = []
    if not forward_only:
        output_tensor_grads = [[] for _ in range(len(model))]

    pipeline_parallel_size = mpu.get_pipeline_model_parallel_world_size()
    pipeline_parallel_rank = mpu.get_pipeline_model_parallel_rank()

    if args.sequence_parallel:
        seq_length = args.seq_length // mpu.get_tensor_model_parallel_world_size()
    else:
        seq_length = args.seq_length
    tensor_shape = (seq_length, args.micro_batch_size, args.hidden_size)
    
    # Compute number of warmup and remaining microbatches.
    num_model_chunks = len(model)
    num_microbatches = get_num_microbatches() * num_model_chunks
    all_warmup_microbatches = False
    if forward_only:
        num_warmup_microbatches = num_microbatches
    else:
        # Run all forward passes and then all backward passes if number of
        # microbatches is just the number of pipeline stages.
        # Otherwise, perform (num_model_chunks-1)*pipeline_parallel_size on
        # all workers, followed by more microbatches after depending on
        # stage ID (more forward passes for earlier stages, later stages can
        # immediately start with 1F1B).
        if get_num_microbatches() == pipeline_parallel_size:
            num_warmup_microbatches = num_microbatches
            all_warmup_microbatches = True
        else:
            num_warmup_microbatches = \
                (pipeline_parallel_size - pipeline_parallel_rank - 1) * 2
            num_warmup_microbatches += (
                num_model_chunks - 1) * pipeline_parallel_size
            num_warmup_microbatches = min(num_warmup_microbatches,
                                          num_microbatches)
    num_microbatches_remaining = \
        num_microbatches - num_warmup_microbatches

    def get_model_chunk_id(microbatch_id, forward):
        """Helper method to get the model chunk ID given the iteration number."""
        microbatch_id_in_group = microbatch_id % (pipeline_parallel_size * num_model_chunks)
        model_chunk_id = microbatch_id_in_group // pipeline_parallel_size
        if not forward:
            model_chunk_id = (num_model_chunks - model_chunk_id - 1)
        return model_chunk_id

    def forward_step_helper(microbatch_id):
        """Helper method to run forward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        forward_step())."""
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=True)
        mpu.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        # forward step
        if mpu.is_pipeline_first_stage():
            if len(input_tensors[model_chunk_id]) == \
                    len(output_tensors[model_chunk_id]):
                input_tensors[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id][-1]
        output_tensor = forward_step(forward_step_func,
                                     data_iterator[model_chunk_id],
                                     model[model_chunk_id],
                                     input_tensor, 
                                     forward_data_store,
                                     timers,
                                     collect_non_loss_data)
        output_tensors[model_chunk_id].append(output_tensor)

        # if forward-only, no need to save tensors for a backward pass
        if forward_only:
            input_tensors[model_chunk_id].pop()
            output_tensors[model_chunk_id].pop()

        return output_tensor

    def backward_step_helper(microbatch_id):
        """Helper method to run backward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        backward_step())."""
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=False)
        mpu.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        if mpu.is_pipeline_last_stage():
            if len(output_tensor_grads[model_chunk_id]) == 0:
                output_tensor_grads[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id].pop(0)
        output_tensor = output_tensors[model_chunk_id].pop(0)
        output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)
        input_tensor_grad = \
            backward_step(optimizer,
                          input_tensor,
                          output_tensor,
                          output_tensor_grad,
                          timers)

        return input_tensor_grad

    # Run warmup forward passes.
    mpu.set_virtual_pipeline_model_parallel_rank(0)
    input_tensors[0].append(
        p2p_communication.recv_forward(tensor_shape, timers=timers))
    for k in range(num_warmup_microbatches):
        output_tensor = forward_step_helper(k)

        # Determine if tensor should be received from previous stage.
        next_forward_model_chunk_id = get_model_chunk_id(k+1, forward=True)
        recv_prev = True
        if mpu.is_pipeline_first_stage(ignore_virtual=True):
            if next_forward_model_chunk_id == 0:
                recv_prev = False
        if k == (num_microbatches - 1):
            recv_prev = False

        # Don't send tensor downstream if on last stage.
        if mpu.is_pipeline_last_stage():
            output_tensor = None

        # If model has LBL, don't send LBL to next rank.
        output_tensor_to_send = output_tensor[0] \
            if (args.model_type == ModelType.encoder_or_decoder_with_lbl and
                output_tensor is not None) \
            else output_tensor

        # Send and receive tensors as appropriate (send tensors computed
        # in this iteration; receive tensors for next iteration).
        if k == (num_warmup_microbatches - 1) and not forward_only and \
                not all_warmup_microbatches:
            input_tensor_grad = None
            recv_next = True
            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                recv_next = False
            input_tensor, output_tensor_grad = \
                p2p_communication.send_forward_backward_recv_forward_backward(
                        output_tensor_to_send, input_tensor_grad,
                        recv_prev=recv_prev, recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        timers=timers)
            output_tensor_grads[num_model_chunks-1].append(output_tensor_grad)
        else:
            input_tensor = \
                p2p_communication.send_forward_recv_forward(
                    output_tensor_to_send, recv_prev=recv_prev,
                    tensor_shape=tensor_shape,
                    timers=timers)
        input_tensors[next_forward_model_chunk_id].append(input_tensor)
        deallocate_output_tensor(output_tensor)

    # Run 1F1B in steady state.
    for k in range(num_microbatches_remaining):
        # Forward pass.
        forward_k = k + num_warmup_microbatches
        output_tensor = forward_step_helper(forward_k)

        # Backward pass.
        backward_k = k
        input_tensor_grad = backward_step_helper(backward_k)

        # Send output_tensor and input_tensor_grad, receive input_tensor
        # and output_tensor_grad.

        # Determine if current stage has anything to send in either direction,
        # otherwise set tensor to None.
        forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
        mpu.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)
        if mpu.is_pipeline_last_stage():
            output_tensor = None

        # If model has LBL, don't send LBL to next rank.
        output_tensor_to_send = output_tensor[0] \
            if (args.model_type == ModelType.encoder_or_decoder_with_lbl and
                output_tensor is not None) \
            else output_tensor

        backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
        mpu.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)
        if mpu.is_pipeline_first_stage():
            input_tensor_grad = None

        # Determine if peers are sending, and where in data structure to put
        # received tensors.
        recv_prev = True
        if mpu.is_pipeline_first_stage(ignore_virtual=True):
            # First stage is ahead of last stage by (pipeline_parallel_size - 1).
            next_forward_model_chunk_id = get_model_chunk_id(
                forward_k - (pipeline_parallel_size - 1), forward=True)
            if next_forward_model_chunk_id == (num_model_chunks - 1):
                recv_prev = False
            next_forward_model_chunk_id += 1
        else:
            next_forward_model_chunk_id = get_model_chunk_id(forward_k + 1,
                                                             forward=True)

        recv_next = True
        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            # Last stage is ahead of first stage by (pipeline_parallel_size - 1).
            next_backward_model_chunk_id = get_model_chunk_id(
                backward_k - (pipeline_parallel_size - 1), forward=False)
            if next_backward_model_chunk_id == 0:
                recv_next = False
            next_backward_model_chunk_id -= 1
        else:
            next_backward_model_chunk_id = get_model_chunk_id(backward_k + 1,
                                                              forward=False)

        # If last iteration, don't receive; we already received one extra
        # before the start of the for loop.
        if k == (num_microbatches_remaining - 1):
            recv_prev = False

        # Communicate tensors.
        input_tensor, output_tensor_grad = \
            p2p_communication.send_forward_backward_recv_forward_backward(
                    output_tensor_to_send, input_tensor_grad,
                    recv_prev=recv_prev, recv_next=recv_next,
                    tensor_shape=tensor_shape, timers=timers)
        deallocate_output_tensor(output_tensor)

        # Put input_tensor and output_tensor_grad in data structures in the
        # right location.
        if recv_prev:
            input_tensors[next_forward_model_chunk_id].append(input_tensor)
        if recv_next:
            output_tensor_grads[next_backward_model_chunk_id].append(
                output_tensor_grad)

    # Run cooldown backward passes (flush out pipeline).
    if not forward_only:
        if all_warmup_microbatches:
            output_tensor_grads[num_model_chunks-1].append(
                p2p_communication.recv_backward(tensor_shape, timers=timers))
        for k in range(num_microbatches_remaining, num_microbatches):
            input_tensor_grad = backward_step_helper(k)
            next_backward_model_chunk_id = get_model_chunk_id(k+1, forward=False)
            recv_next = True
            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                if next_backward_model_chunk_id == (num_model_chunks - 1):
                    recv_next = False
            if k == (num_microbatches - 1):
                recv_next = False
            output_tensor_grads[next_backward_model_chunk_id].append(
                p2p_communication.send_backward_recv_backward(
                    input_tensor_grad, recv_next=recv_next,
                    tensor_shape=tensor_shape,
                    timers=timers))

    return forward_data_store


def get_tensor_shapes(rank, model_type):
    # Determine right tensor sizes (based on position of rank with respect to split
    # rank) and model size.
    # Send two tensors if model is T5 and rank is in decoder stage:
    #     first tensor is decoder (pre-transpose),
    #     second tensor is encoder (post-transpose).
    # If model is T5 and rank is at the boundary:
    #     send one tensor (post-transpose from encoder).
    # Otherwise, send one tensor (pre-transpose).
    args = get_args()
    tensor_shapes = []

    if args.sequence_parallel:
        seq_length = args.seq_length // mpu.get_tensor_model_parallel_world_size()
    else:
        seq_length = args.seq_length

    if model_type == ModelType.encoder_and_decoder:
        if args.sequence_parallel:
            decoder_seq_length = args.decoder_seq_length // mpu.get_tensor_model_parallel_world_size()
        else:
            decoder_seq_length = args.decoder_seq_length

        if mpu.is_pipeline_stage_before_split(rank):
            tensor_shapes.append((seq_length, args.micro_batch_size, args.hidden_size))
        else:
            tensor_shapes.append((decoder_seq_length, args.micro_batch_size, args.hidden_size))
            tensor_shapes.append((seq_length, args.micro_batch_size, args.hidden_size))
    else:
        tensor_shapes.append((seq_length, args.micro_batch_size, args.hidden_size))
    return tensor_shapes


def recv_forward(tensor_shapes, timers):
    input_tensors = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            input_tensors.append(None)
        else:
            input_tensors.append(p2p_communication.recv_forward(tensor_shape,
                                                                timers=timers))
    return input_tensors


def recv_backward(tensor_shapes, timers):
    output_tensor_grads = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            output_tensor_grads.append(None)
        else:
            output_tensor_grads.append(p2p_communication.recv_backward(tensor_shape,
                                                                       timers=timers))
    return output_tensor_grads


def send_forward(output_tensors, tensor_shapes, timers):
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    for (output_tensor, tensor_shape) in zip(output_tensors, tensor_shapes):
        if tensor_shape is None:
            continue
        p2p_communication.send_forward(output_tensor, tensor_shape, timers=timers)


def send_backward(input_tensor_grads, tensor_shapes, timers):
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
    for (input_tensor_grad, tensor_shape) in zip(input_tensor_grads, tensor_shapes):
        if tensor_shape is None:
            continue
        p2p_communication.send_backward(input_tensor_grad, tensor_shape, timers=timers)


def send_forward_recv_backward(output_tensors, tensor_shapes, timers):
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    output_tensor_grads = []
    for (output_tensor, tensor_shape) in zip(output_tensors, tensor_shapes):
        if tensor_shape is None:
            output_tensor_grads.append(None)
            continue
        output_tensor_grad = p2p_communication.send_forward_recv_backward(
                output_tensor, tensor_shape, timers=timers)
        output_tensor_grads.append(output_tensor_grad)
    return output_tensor_grads


def send_backward_recv_forward(input_tensor_grads, tensor_shapes, timers):
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
    input_tensors = []
    for (input_tensor_grad, tensor_shape) in zip(input_tensor_grads, tensor_shapes):
        if tensor_shape is None:
            input_tensors.append(None)
            continue
        input_tensor = p2p_communication.send_backward_recv_forward(
                input_tensor_grad, tensor_shape, timers=timers)
        input_tensors.append(input_tensor)
    return input_tensors


def forward_backward_pipelining_without_interleaving(forward_step_func,
                                                     data_iterator,
                                                     model,
                                                     optimizer,
                                                     timers,
                                                     forward_only,
                                                     collect_non_loss_data=False):
    """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages.

    Returns dictionary with losses if the last stage, empty dict otherwise."""
    args = get_args()
    
    assert len(model) == 1
    model = model[0]

    # Compute number of warmup microbatches.
    num_microbatches = get_num_microbatches()
    num_warmup_microbatches = \
        (mpu.get_pipeline_model_parallel_world_size() -
         mpu.get_pipeline_model_parallel_rank() - 1)
    num_warmup_microbatches = min(
        num_warmup_microbatches,
        num_microbatches)
    num_microbatches_remaining = \
        num_microbatches - num_warmup_microbatches

    unwrapped_model = unwrap_model(
        model, (torchDDP, LocalDDP, Float16Module))
    model_type = unwrapped_model.model_type
    rank = mpu.get_pipeline_model_parallel_rank()
    recv_tensor_shapes = get_tensor_shapes(rank-1, model_type)
    send_tensor_shapes = get_tensor_shapes(rank, model_type)

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    output_tensors = None
    if not forward_only:
        input_tensors = []
        output_tensors = []
    forward_data_store = []

    # Run warmup forward passes.
    for i in range(num_warmup_microbatches):
        input_tensor = recv_forward(recv_tensor_shapes, timers=timers)
        output_tensor = forward_step(forward_step_func, data_iterator, model,
                                     input_tensor, forward_data_store,
                                     timers, collect_non_loss_data)

        # If model has LBL, don't send LBL to next rank.
        output_tensor_to_send = output_tensor[0] \
            if args.model_type == ModelType.encoder_or_decoder_with_lbl \
            else output_tensor
        send_forward(output_tensor_to_send, send_tensor_shapes, timers=timers)

        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor)

    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    if num_microbatches_remaining > 0:
        input_tensor = recv_forward(recv_tensor_shapes, timers=timers)

    # Run 1F1B in steady state.
    for i in range(num_microbatches_remaining):
        last_iteration = (i == (num_microbatches_remaining - 1))

        output_tensor = forward_step(forward_step_func, data_iterator, model,
                                     input_tensor, forward_data_store,
                                     timers, collect_non_loss_data)

        # If model has LBL, don't send LBL to next rank.
        output_tensor_to_send = output_tensor[0] \
            if args.model_type == ModelType.encoder_or_decoder_with_lbl \
            else output_tensor
        if forward_only:
            send_forward(output_tensor_to_send, send_tensor_shapes, timers=timers)

            if not last_iteration:
                input_tensor = recv_forward(recv_tensor_shapes, timers=timers)

        else:
            output_tensor_grad = \
                send_forward_recv_backward(output_tensor_to_send,
                                           send_tensor_shapes,
                                           timers=timers)

            # Add input_tensor and output_tensor to end of list.
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor)

            # Pop input_tensor and output_tensor from the start of the list for
            # the backward pass.
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            input_tensor_grad = \
                backward_step(optimizer, input_tensor, output_tensor,
                              output_tensor_grad, timers)

            if last_iteration:
                input_tensor = None
                send_backward(input_tensor_grad, recv_tensor_shapes, timers=timers)
            else:
                input_tensor = \
                    send_backward_recv_forward(
                        input_tensor_grad, recv_tensor_shapes, timers=timers)

    # Run cooldown backward passes.
    if not forward_only:
        for i in range(num_warmup_microbatches):
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            output_tensor_grad = recv_backward(send_tensor_shapes, timers=timers)

            input_tensor_grad = \
                backward_step(optimizer, input_tensor, output_tensor,
                              output_tensor_grad, timers)

            send_backward(input_tensor_grad, recv_tensor_shapes, timers=timers)

    return forward_data_store
