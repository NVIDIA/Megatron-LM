# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pretrain utilities."""

from datetime import datetime
import math
import sys
import time
# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()

import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron import get_args
from megatron import get_timers
from megatron import get_tensorboard_writer
from megatron import get_current_global_batch_size
from megatron import get_num_microbatches
from megatron import is_last_rank
from megatron import update_num_microbatches
from megatron import mpu
from megatron import print_rank_0
from megatron import print_rank_last
from megatron.checkpointing import load_checkpoint
from megatron.checkpointing import save_checkpoint
from megatron.model import FP16Module
from megatron.optimizer import get_megatron_optimizer

from megatron.initialize import initialize_megatron
from megatron.initialize import write_args_to_tensorboard
from megatron.learning_rates import AnnealingLR
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model.realm_model import ICTBertModel
from megatron.utils import check_adlr_autoresume_termination
from megatron.data.data_loaders import build_pretraining_data_loader
from megatron.utils import report_memory


def print_datetime(string):
    """Note that this call will sync across all ranks."""
    torch.distributed.barrier()
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print_rank_0('[' + string + '] datetime: {} '.format(time_str))


def pretrain(train_valid_test_dataset_provider, model_provider,
             forward_step_func, extra_args_provider=None, args_defaults={}):
    """Main training program.

    This function will run the followings in the order provided:
        1) initialize Megatron.
        2) setup model, optimizer and lr schedule using the model_provider.
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the modle using the forward_step_func.

    Arguments:
        train_valid_test_dataset_provider: a function that takes the size of
            train/valid/test dataset and returns `train, valid, test` datasets.
        model_provider: a function that returns a vanilla version of the
            model. By vanilla we mean a simple model on cpu with no fp16 or ddp.
        forward_step_func: a function that takes a `data iterator` and `model`,
            and returns a `loss` scalar with a dictionary with key:values being
            the info we would like to monitor during training, for example
            `lm-loss: value`. We also require that this function add
            `batch generator` to the timers class.
        extra_args_provider: a function that takes a parser and adds arguments
            to it. It is used for programs to add their own arguments.
        args_defaults: a dictionary from argument-name to argument-value. It
            to set already parse arguments.
    """

    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults)

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    global _TRAIN_START_TIME
    start_time_tensor = torch.cuda.FloatTensor([_TRAIN_START_TIME])
    torch.distributed.all_reduce(start_time_tensor,
                                 op=torch.distributed.ReduceOp.MIN)
    _TRAIN_START_TIME = start_time_tensor.item()
    print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(
        time.time() - _TRAIN_START_TIME))
    print_datetime('after megatron is initialized')

    args = get_args()
    timers = get_timers()

    # Model, optimizer, and learning rate.
    timers('model and optimizer').start()
    model, optimizer, lr_scheduler = setup_model_and_optimizer(model_provider)
    timers('model and optimizer').stop()
    print_datetime('after model, optimizer, and learning rate '
                   'scheduler are built')

    # Data stuff.
    timers('train/valid/test data iterators').start()
    train_data_iterator, valid_data_iterator, test_data_iterator \
        = build_train_valid_test_data_iterators(
            train_valid_test_dataset_provider)
    timers('train/valid/test data iterators').stop()
    print_datetime('after dataloaders are built')

    # Print setup timing.
    print_rank_0('done with setups ...')
    timers.log(['model and optimizer', 'train/valid/test data iterators'])
    print_rank_0('training ...')

    iteration = 0
    if args.do_train and args.train_iters > 0:
        iteration = train(forward_step_func,
                          model, optimizer, lr_scheduler,
                          train_data_iterator, valid_data_iterator)
    print_datetime('after training is done')

    if args.do_valid:
        prefix = 'the end of training for val data'
        evaluate_and_print_results(prefix, forward_step_func,
                                   valid_data_iterator, model,
                                   iteration, False)

    if args.save and iteration != 0:
        save_checkpoint(iteration, model, optimizer, lr_scheduler)

    if args.do_test:
        # Run on test data.
        prefix = 'the end of training for test data'
        evaluate_and_print_results(prefix, forward_step_func,
                                   test_data_iterator, model,
                                   0, True)

def update_train_iters(args):

    # For iteration-based training, we don't need to do anything
    if args.train_iters:
        return

    # Constant batch size with sample-based training.
    if args.rampup_batch_size is None:
        args.train_iters = args.train_samples // args.global_batch_size

    else:
        # Sample based training with rampup batch size.
        iterations = 0
        consumed_samples = 0
        # Rampup phase.
        while consumed_samples <= int(args.rampup_batch_size[2]):
            update_num_microbatches(consumed_samples, consistency_check=False)
            consumed_samples += get_current_global_batch_size()
            iterations += 1
        # Reset
        update_num_microbatches(0, consistency_check=False)
        # Constant phase
        # Note that we throw away any partial last batch.
        iterations += (args.train_samples - consumed_samples) // \
                      args.global_batch_size
        args.train_iters = iterations

    print_rank_0('setting training iterations to {}'.format(args.train_iters))


def get_model(model_provider_func):
    """Build the model."""
    args = get_args()

    # Build model on cpu.
    model = model_provider_func()

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for param in model.parameters():
        mpu.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on (tensor, pipeline) '
              'model parallel rank ({}, {}): {}'.format(
            mpu.get_tensor_model_parallel_rank(),
            mpu.get_pipeline_model_parallel_rank(),
            sum([p.nelement() for p in model.parameters()])), flush=True)

    # GPU allocation.
    model.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16:
        model = FP16Module(model)

    if args.DDP_impl == 'torch':
        i = torch.cuda.current_device()
        model = torchDDP(model, device_ids=[i], output_device=i,
                         process_group=mpu.get_data_parallel_group())
        return model
    if args.DDP_impl == 'local':
        model = LocalDDP(model)
        return model

    raise NotImplementedError('Unknown DDP implementation specified: {}. '
                              'Exiting.'.format(args.DDP_impl))


def get_learning_rate_scheduler(optimizer):
    """Build the learning rate scheduler."""
    args = get_args()

    # Iteration-based training.
    if args.train_iters:
        if args.lr_decay_iters is None:
            args.lr_decay_iters = args.train_iters
        decay_steps = args.lr_decay_iters * args.global_batch_size
        if args.lr_warmup_fraction is not None:
            warmup_steps = args.lr_warmup_fraction * decay_steps
        else:
            warmup_steps = args.lr_warmup_iters * args.global_batch_size
    # Sample-based training.
    elif args.train_samples:
        # We need to set training iters for later use. Technically
        # we need to adjust the training samples too (due to last
        # batch being incomplete) but we leave it as is for now.
        update_train_iters(args)
        if args.lr_decay_samples is None:
            args.lr_decay_samples = args.train_samples
        decay_steps = args.lr_decay_samples
        if args.lr_warmup_fraction is not None:
            warmup_steps = args.lr_warmup_fraction * decay_steps
        else:
            warmup_steps = args.lr_warmup_samples
    else:
        raise Exception(
            'either train-iters or train-samples should be provided.')

    lr_scheduler = AnnealingLR(
        optimizer,
        max_lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        decay_style=args.lr_decay_style,
        use_checkpoint_lr_scheduler=args.use_checkpoint_lr_scheduler,
        override_lr_scheduler=args.override_lr_scheduler)

    return lr_scheduler


def setup_model_and_optimizer(model_provider_func):
    """Setup model and optimizer."""
    args = get_args()

    model = get_model(model_provider_func)

    unwrapped_model = model
    while isinstance(unwrapped_model, (torchDDP, LocalDDP, FP16Module)):
        unwrapped_model = unwrapped_model.module
    optimizer = get_megatron_optimizer(unwrapped_model)

    lr_scheduler = get_learning_rate_scheduler(optimizer)

    if args.load is not None:
        timers = get_timers()
        # Extra barrier is added to make sure all ranks report the
        # max time.
        torch.distributed.barrier()
        timers('load checkpoint').start()
        args.iteration = load_checkpoint(model, optimizer, lr_scheduler)
        torch.distributed.barrier()
        timers('load checkpoint').stop()
        timers.log(['load checkpoint'])
    else:
        args.iteration = 0

    # We only support local DDP with multiple micro-batches.
    if get_num_microbatches() > 1:
        assert args.DDP_impl == 'local'

    # get model without FP16 and/or TorchDDP wrappers
    unwrapped_model = model
    while hasattr(unwrapped_model, 'module'):
        unwrapped_model = unwrapped_model.module

    if args.iteration == 0 and hasattr(unwrapped_model,
                                       'init_state_dict_from_bert'):
        print("Initializing ICT from pretrained BERT model", flush=True)
        unwrapped_model.init_state_dict_from_bert()

    return model, optimizer, lr_scheduler


def communicate(tensor_send_next, tensor_send_prev, recv_forward, recv_backward):
    """Communicate tensors between stages."""
    args = get_args()

    # Create placeholder tensors for receive in forward and backward directions
    # if needed.
    tensor_recv_prev = None
    tensor_recv_next = None
    tensor_shape = (args.seq_length, args.micro_batch_size, args.hidden_size)
    dtype = args.params_dtype
    if args.fp32_residual_connection:
        dtype = torch.float
    if recv_forward:
        tensor_recv_prev = torch.empty(tensor_shape,
                                       requires_grad=True,
                                       device=torch.cuda.current_device(),
                                       dtype=dtype)
    if recv_backward:
        tensor_recv_next = torch.empty(tensor_shape,
                                       requires_grad=True,
                                       device=torch.cuda.current_device(),
                                       dtype=dtype)

    # Send tensors in both the forward and backward directions as appropriate.
    ops = []
    if tensor_send_prev is not None:
        send_prev_op = torch.distributed.P2POp(torch.distributed.isend, tensor_send_prev,
                                               mpu.get_pipeline_model_parallel_prev_rank())
        ops.append(send_prev_op)
    if tensor_recv_prev is not None:
        recv_prev_op = torch.distributed.P2POp(torch.distributed.irecv, tensor_recv_prev,
                                               mpu.get_pipeline_model_parallel_prev_rank())
        ops.append(recv_prev_op)
    if tensor_send_next is not None:
        send_next_op = torch.distributed.P2POp(torch.distributed.isend, tensor_send_next,
                                               mpu.get_pipeline_model_parallel_next_rank())
        ops.append(send_next_op)
    if tensor_recv_next is not None:
        recv_next_op = torch.distributed.P2POp(torch.distributed.irecv, tensor_recv_next,
                                               mpu.get_pipeline_model_parallel_next_rank())
        ops.append(recv_next_op)
    reqs = torch.distributed.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()

    return tensor_recv_prev, tensor_recv_next


def backward_step(optimizer, model, input_tensor, output_tensor, output_tensor_grad):
    """Backward step."""
    args = get_args()
    timers = get_timers()

    # Retain the grad on the input_tensor.
    if input_tensor is not None:
        input_tensor.retain_grad()

    # Backward pass.
    if output_tensor_grad is None:
        output_tensor = optimizer.scale_loss(output_tensor)
    torch.autograd.backward(output_tensor, grad_tensors=output_tensor_grad)

    # Collect the grad of the input_tensor.
    input_tensor_grad = None
    if input_tensor is not None:
        input_tensor_grad = input_tensor.grad

    return input_tensor_grad


def forward_step_with_communication(forward_step_func, data_iterator, model,
                                    input_tensors, output_tensors,
                                    losses_reduced, timers):
    args = get_args()

    if not mpu.is_pipeline_first_stage():
        timers('forward-recv').start()
        input_tensor, _ = communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_forward=True,
            recv_backward=False)
        timers('forward-recv').stop()
    else:
        input_tensor = None

    # Forward model for one step.
    timers('forward-compute').start()
    output_tensor = forward_step_func(data_iterator, model, input_tensor)
    timers('forward-compute').stop()

    if mpu.is_pipeline_last_stage():
        loss, loss_reduced = output_tensor
        output_tensor = loss / get_num_microbatches()
        losses_reduced.append(loss_reduced)
    else:
        timers('forward-send').start()
        communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_forward=False,
            recv_backward=False)
        timers('forward-send').stop()

    input_tensors.append(input_tensor)
    output_tensors.append(output_tensor)


def backward_step_with_communication(optimizer, model, input_tensors, output_tensors, timers):
    input_tensor = input_tensors.pop(0)
    output_tensor = output_tensors.pop(0)

    if mpu.is_pipeline_last_stage():
        output_tensor_grad = None
    else:
        timers('backward-recv').start()
        _, output_tensor_grad = communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_forward=False,
            recv_backward=True)
        timers('backward-recv').stop()

    # Backward pass for one step.
    timers('backward-compute').start()
    input_grad_tensor = \
        backward_step(optimizer, model, input_tensor, output_tensor, output_tensor_grad)
    timers('backward-compute').stop()

    if not mpu.is_pipeline_first_stage():
        timers('backward-send').start()
        communicate(
            tensor_send_next=None,
            tensor_send_prev=input_grad_tensor,
            recv_forward=False,
            recv_backward=False)
        timers('backward-send').stop()


def forward_and_backward_steps_with_communication(forward_step_func, data_iterator, model,
                                                  optimizer,
                                                  input_tensor, last_microbatch,
                                                  input_tensors, output_tensors,
                                                  losses_reduced, timers):
    args = get_args()

    # Forward model for one step.
    timers('forward-compute').start()
    output_tensor = forward_step_func(data_iterator, model, input_tensor)
    timers('forward-compute').stop()

    if mpu.is_pipeline_last_stage():
        loss, loss_reduced = output_tensor
        output_tensor = loss / get_num_microbatches()
        output_tensor_grad = None
        losses_reduced.append(loss_reduced)
    else:
        timers('forward-send-backward-recv').start()
        _, output_tensor_grad = communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_forward=False,
            recv_backward=True)
        timers('forward-send-backward-recv').stop()

    input_tensors.append(input_tensor)
    output_tensors.append(output_tensor)

    input_tensor = input_tensors.pop(0)
    output_tensor = output_tensors.pop(0)

    # Backward pass for one step.
    timers('backward-compute').start()
    input_grad_tensor = \
        backward_step(optimizer, model, input_tensor, output_tensor, output_tensor_grad)
    timers('backward-compute').stop()

    if not mpu.is_pipeline_first_stage():
        timers('backward-send-forward-recv').start()
        input_tensor, _ = communicate(
            tensor_send_next=None,
            tensor_send_prev=input_grad_tensor,
            recv_forward=(not last_microbatch),
            recv_backward=False)
        timers('backward-send-forward-recv').stop()
    else:
        input_tensor = None

    return input_tensor


def forward_backward_no_pipelining(forward_step_func, data_iterator, model,
                                   optimizer, timers):
    """Run forward and backward passes without inter-stage communication."""
    args = get_args()

    losses_reduced = []
    for i in range(get_num_microbatches()):
        timers('forward-compute').start()
        loss, loss_reduced = forward_step_func(data_iterator, model, input_tensor=None)
        output_tensor = loss / get_num_microbatches()
        losses_reduced.append(loss_reduced)
        timers('forward-compute').stop()

        timers('backward-compute').start()
        output_tensor_grad = None
        backward_step(optimizer, model, input_tensor=None,
                      output_tensor=output_tensor, output_tensor_grad=None)
        timers('backward-compute').stop()

    return losses_reduced


def forward_backward_pipelining(forward_step_func, data_iterator, model,
                                optimizer, timers):
    """Run 1F1B schedule, with communication and warmup + cooldown microbatches as needed."""
    args = get_args()

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

    input_tensors = []
    output_tensors = []
    losses_reduced = []

    # Run warmup forward passes.
    for i in range(num_warmup_microbatches):
        forward_step_with_communication(
            forward_step_func, data_iterator, model,
            input_tensors, output_tensors,
            losses_reduced, timers)

    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    if num_microbatches_remaining > 0:
        if mpu.is_pipeline_first_stage():
            input_tensor = None
        else:
            timers('forward-recv').start()
            input_tensor, _ = communicate(tensor_send_next=None,
                                          tensor_send_prev=None,
                                          recv_forward=True,
                                          recv_backward=False)
            timers('forward-recv').stop()

    # Run 1F1B.
    for i in range(num_microbatches_remaining):
        last_iteration = (i == (num_microbatches_remaining - 1))
        input_tensor = \
            forward_and_backward_steps_with_communication(forward_step_func, data_iterator, model,
                                                          optimizer,
                                                          input_tensor, last_iteration,
                                                          input_tensors, output_tensors,
                                                          losses_reduced, timers)

    # Run cooldown backward passes.
    for i in range(num_warmup_microbatches):
        backward_step_with_communication(
            optimizer, model, input_tensors, output_tensors, timers)

    return losses_reduced


def train_step(forward_step_func, data_iterator,
               model, optimizer, lr_scheduler):
    """Single training step."""
    args = get_args()
    timers = get_timers()

    # Set grad to zero.
    optimizer.zero_grad()

    if mpu.get_pipeline_model_parallel_world_size() > 1:
        losses_reduced = forward_backward_pipelining(
            forward_step_func, data_iterator, model, optimizer, timers)
    else:
        losses_reduced = forward_backward_no_pipelining(
            forward_step_func, data_iterator, model, optimizer, timers)

    # All-reduce if needed.
    if args.DDP_impl == 'local':
        timers('backward-params-all-reduce').start()
        model.allreduce_params(reduce_after=False,
                               fp32_allreduce=args.fp32_allreduce)
        timers('backward-params-all-reduce').stop()

    # All-reduce word_embeddings' grad across first and last stages to ensure
    # that word_embeddings parameters stay in sync.
    # This should only run for models that support pipelined model parallelism
    # (BERT and GPT-2).
    timers('backward-embedding-all-reduce').start()
    if (mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()) and \
            mpu.get_pipeline_model_parallel_world_size() > 1:
        unwrapped_model = model
        while isinstance(unwrapped_model, (torchDDP, LocalDDP, FP16Module)):
            unwrapped_model = unwrapped_model.module

        if unwrapped_model.share_word_embeddings:
            word_embeddings_weight = unwrapped_model.word_embeddings_weight()
            torch.distributed.all_reduce(word_embeddings_weight.grad,
                                         group=mpu.get_embedding_group())
    timers('backward-embedding-all-reduce').stop()

    # Update parameters.
    timers('optimizer').start()
    update_successfull = optimizer.step()
    timers('optimizer').stop()

    # Update learning rate.
    if update_successfull:
        increment = get_num_microbatches() * \
                    args.micro_batch_size * \
                    args.data_parallel_size
        lr_scheduler.step(increment=increment)
        skipped_iter = 0
    else:
        skipped_iter = 1

    if mpu.is_pipeline_last_stage():
        # Average loss across microbatches.
        loss_reduced = {}
        for key in losses_reduced[0]:
            losses_reduced_for_key = [x[key] for x in losses_reduced]
            loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
        return loss_reduced, skipped_iter
    return {}, skipped_iter


def training_log(loss_dict, total_loss_dict, learning_rate, iteration,
                 loss_scale, report_memory_flag, skipped_iter):
    """Log training information such as losses, timing, ...."""
    args = get_args()
    timers = get_timers()
    writer = get_tensorboard_writer()

    # Advanced, skipped, and Nan iterations.
    advanced_iters_key = 'advanced iterations'
    skipped_iters_key = 'skipped iterations'
    nan_iters_key = 'nan iterations'
    # Advanced iterations.
    if not skipped_iter:
        total_loss_dict[advanced_iters_key] = total_loss_dict.get(
            advanced_iters_key, 0) + 1
    else:
        if advanced_iters_key not in total_loss_dict:
            total_loss_dict[advanced_iters_key] = 0
    # Skipped iterations.
    total_loss_dict[skipped_iters_key] = total_loss_dict.get(
        skipped_iters_key, 0) + skipped_iter
    # Update losses and set nan iterations
    got_nan = False
    for key in loss_dict:
        if not skipped_iter:
            total_loss_dict[key] = total_loss_dict.get(
                key, torch.cuda.FloatTensor([0.0])) + loss_dict[key]
        else:
            value = loss_dict[key].float().sum().item()
            is_nan = value == float('inf') or \
                     value == -float('inf') or \
                     value != value
            got_nan = got_nan or is_nan
    total_loss_dict[nan_iters_key] = total_loss_dict.get(
        nan_iters_key, 0) + int(got_nan)

    # Logging.
    timers_to_log = []

    def add_to_logging(name):
        if name in timers.timers:
            timers_to_log.append(name)
    add_to_logging('forward-compute')
    add_to_logging('forward-recv')
    add_to_logging('forward-send')
    add_to_logging('forward-send-backward-recv')
    add_to_logging('backward-compute')
    add_to_logging('backward-recv')
    add_to_logging('backward-send')
    add_to_logging('backward-send-forward-recv')
    add_to_logging('backward-params-all-reduce')
    add_to_logging('backward-embedding-all-reduce')
    add_to_logging('optimizer-copy-to-main-grad')
    add_to_logging('optimizer-unscale-and-check-inf')
    add_to_logging('optimizer-clip-main-grad')
    add_to_logging('optimizer-copy-main-to-model-params')
    add_to_logging('optimizer')
    add_to_logging('batch-generator')

    # Calculate batch size.
    batch_size = args.micro_batch_size * args.data_parallel_size * \
        get_num_microbatches()

    total_iterations = total_loss_dict[advanced_iters_key] + \
                       total_loss_dict[skipped_iters_key]

    # Tensorboard values.
    if writer and is_last_rank():
        writer.add_scalar('learning-rate', learning_rate, iteration)
        writer.add_scalar('learning-rate vs samples', learning_rate,
                          args.consumed_train_samples)
        writer.add_scalar('batch-size', batch_size, iteration)
        writer.add_scalar('batch-size vs samples', batch_size,
                          args.consumed_train_samples)
        for key in loss_dict:
            writer.add_scalar(key , loss_dict[key], iteration)
            writer.add_scalar(key + ' vs samples', loss_dict[key],
                              args.consumed_train_samples)
        writer.add_scalar('loss-scale', loss_scale, iteration)
        writer.add_scalar('loss-scale vs samples', loss_scale,
                          args.consumed_train_samples)
        timers.write(timers_to_log, writer, iteration,
                     normalizer=total_iterations)

    if iteration % args.log_interval == 0:
        elapsed_time = timers('interval time').elapsed()
        elapsed_time_per_iteration = elapsed_time / total_iterations
        if writer and torch.distributed.get_rank() == 0:
            writer.add_scalar('iteration-time',
                              elapsed_time_per_iteration, iteration)
        log_string = ' iteration {:8d}/{:8d} |'.format(
            iteration, args.train_iters)
        log_string += ' consumed samples: {:12d} |'.format(
            args.consumed_train_samples)
        log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
            elapsed_time_per_iteration * 1000.0)
        log_string += ' learning rate: {:.3E} |'.format(learning_rate)
        log_string += ' global batch size: {:5d} |'.format(batch_size)
        for key in total_loss_dict:
            if key not in [advanced_iters_key, skipped_iters_key,
                           nan_iters_key]:
                avg = total_loss_dict[key].item() / \
                      float(max(1, total_loss_dict[advanced_iters_key]))
                if avg > 0.0:
                    log_string += ' {}: {:.6E} |'.format(key, avg)
                total_loss_dict[key] = torch.cuda.FloatTensor([0.0])
        log_string += ' loss scale: {:.1f} |'.format(loss_scale)
        log_string += ' number of skipped iterations: {:3d} |'.format(
            total_loss_dict[skipped_iters_key])
        log_string += ' number of nan iterations: {:3d} |'.format(
            total_loss_dict[nan_iters_key])
        total_loss_dict[advanced_iters_key] = 0
        total_loss_dict[skipped_iters_key] = 0
        total_loss_dict[nan_iters_key] = 0
        print_rank_last(log_string)
        if report_memory_flag and learning_rate > 0.:
            # Report memory after optimizer state has been initialized.
            report_memory('(after {} iterations)'.format(iteration))
            report_memory_flag = False
        timers.log(timers_to_log, normalizer=args.log_interval)

    return report_memory_flag


def save_checkpoint_and_time(iteration, model, optimizer, lr_scheduler):
    timers = get_timers()
    # Extra barrier is added to make sure
    # all ranks report the max time.
    torch.distributed.barrier()
    timers('save checkpoint').start()
    save_checkpoint(iteration, model, optimizer, lr_scheduler)
    torch.distributed.barrier()
    timers('save checkpoint').stop()
    timers.log(['save checkpoint'])


def train(forward_step_func, model, optimizer, lr_scheduler,
          train_data_iterator, valid_data_iterator):
    """Train the model function."""
    args = get_args()
    timers = get_timers()

    # Write args to tensorboard
    write_args_to_tensorboard()

    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    total_loss_dict = {}

    # Iterations.
    iteration = args.iteration

    timers('interval time').start()
    print_datetime('before the start of training step')
    report_memory_flag = True
    while iteration < args.train_iters:
        update_num_microbatches(args.consumed_train_samples)
        loss_dict, skipped_iter = train_step(forward_step_func,
                                             train_data_iterator,
                                             model,
                                             optimizer,
                                             lr_scheduler)
        iteration += 1
        args.consumed_train_samples += mpu.get_data_parallel_world_size() * \
                                       args.micro_batch_size * \
                                       get_num_microbatches()

        # Logging.
        loss_scale = optimizer.get_loss_scale().item()
        report_memory_flag = training_log(loss_dict, total_loss_dict,
                                          optimizer.param_groups[0]['lr'],
                                          iteration, loss_scale,
                                          report_memory_flag, skipped_iter)

        # Autoresume
        if args.adlr_autoresume and \
           (iteration % args.adlr_autoresume_interval == 0):
            check_adlr_autoresume_termination(iteration, model, optimizer,
                                              lr_scheduler)

        # Evaluation
        if args.eval_interval and iteration % args.eval_interval == 0 and \
           args.do_valid:
            prefix = 'iteration {}'.format(iteration)
            evaluate_and_print_results(prefix, forward_step_func,
                                       valid_data_iterator, model,
                                       iteration, False)

        # Checkpointing
        saved_checkpoint = False
        if args.save and args.save_interval and \
           iteration % args.save_interval == 0:
            save_checkpoint_and_time(iteration, model, optimizer,
                                     lr_scheduler)
            saved_checkpoint = True

        # Exiting based on duration
        if args.exit_duration_in_mins:
            train_time = (time.time() - _TRAIN_START_TIME) / 60.0
            done_cuda = torch.cuda.IntTensor(
                [train_time > args.exit_duration_in_mins])
            torch.distributed.all_reduce(
                done_cuda, op=torch.distributed.ReduceOp.MAX)
            done = done_cuda.item()
            if done:
                if not saved_checkpoint:
                    save_checkpoint_and_time(iteration, model, optimizer,
                                             lr_scheduler)
                print_datetime('exiting program after {} minutes'.format(train_time))                
                sys.exit()

        # Exiting based on iterations        
        if args.exit_interval and iteration % args.exit_interval == 0:
            if not saved_checkpoint:
                save_checkpoint_and_time(iteration, model, optimizer,
                                         lr_scheduler)
            torch.distributed.barrier()
            print_datetime('exiting program at iteration {}'.format(iteration))                
            sys.exit()


    return iteration


def evaluate(forward_step_func, data_iterator, model, verbose=False):
    """Evaluation."""
    args = get_args()

    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_loss_dict = {}

    with torch.no_grad():
        iteration = 0
        while iteration < args.eval_iters:
            iteration += 1
            if verbose and iteration % args.log_interval == 0:
                print_rank_0('Evaluating iter {}/{}'.format(iteration,
                                                            args.eval_iters))

            for _ in range(get_num_microbatches()):
                if not mpu.is_pipeline_first_stage():
                    input_tensor, _ = communicate(
                        tensor_send_next=None,
                        tensor_send_prev=None,
                        recv_forward=True,
                        recv_backward=False)
                else:
                    input_tensor = None

                # Forward evaluation.
                output_tensor = forward_step_func(data_iterator, model, input_tensor)

                if mpu.is_pipeline_last_stage():
                    _, loss_dict = output_tensor
                    # Reduce across processes.
                    for key in loss_dict:
                        total_loss_dict[key] = total_loss_dict.get(key, torch.cuda.FloatTensor([0.0])) + \
                            loss_dict[key]
                else:
                    communicate(
                        tensor_send_next=output_tensor,
                        tensor_send_prev=None,
                        recv_forward=False,
                        recv_backward=False)

            args.consumed_valid_samples += mpu.get_data_parallel_world_size() \
                                           * args.micro_batch_size \
                                           * get_num_microbatches()
    # Move model back to the train mode.
    model.train()

    for key in total_loss_dict:
        total_loss_dict[key] /= args.eval_iters * get_num_microbatches()

    return total_loss_dict

def evaluate_and_print_results(prefix, forward_step_func,
                               data_iterator, model,
                               iteration, verbose=False):
    """Helper function to evaluate and dump results on screen."""
    args = get_args()
    writer = get_tensorboard_writer()

    total_loss_dict = evaluate(forward_step_func, data_iterator, model, verbose)
    string = ' validation loss at {} | '.format(prefix)
    for key in total_loss_dict:
        string += '{} value: {:.6E} | '.format(key, total_loss_dict[key].item())
        ppl = math.exp(min(20, total_loss_dict[key].item()))
        string += '{} PPL: {:.6E} | '.format(key, ppl)
        if writer and is_last_rank():
            writer.add_scalar('{} value-validation'.format(key),
                              total_loss_dict[key].item(),
                              iteration)
            writer.add_scalar('{} ppl-validation'.format(key), ppl, iteration)
            writer.add_scalar('{} value-validation vs samples'.format(key),
                              total_loss_dict[key].item(),
                              args.consumed_train_samples)
            writer.add_scalar('{} ppl-validation vs samples'.format(key), ppl,
                              args.consumed_train_samples)

    length = len(string) + 1
    print_rank_last('-' * length)
    print_rank_last(string)
    print_rank_last('-' * length)


def build_train_valid_test_data_iterators(
        build_train_valid_test_datasets_provider):
    """XXX"""
    args = get_args()

    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

    print_rank_0('> building train, validation, and test datasets ...')

    # Backward compatibility, assume fixed batch size.
    if args.iteration > 0 and args.consumed_train_samples == 0:
        assert args.train_samples is None, \
            'only backward compatiblity support for iteration-based training'
        args.consumed_train_samples = args.iteration * args.global_batch_size
    if args.iteration > 0 and args.consumed_valid_samples == 0:
        assert args.train_samples is None, \
            'only backward compatiblity support for iteration-based training'
        args.consumed_valid_samples = (args.iteration // args.eval_interval) * \
            args.eval_iters * args.global_batch_size

    # Data loader only on rank 0 of each model parallel group.
    if mpu.get_tensor_model_parallel_rank() == 0:

        # Number of train/valid/test samples.
        if args.train_samples:
            train_samples = args.train_samples
        else:
            train_samples = args.train_iters * args.global_batch_size
        eval_iters = (args.train_iters // args.eval_interval + 1) * \
                     args.eval_iters
        test_iters = args.eval_iters
        train_val_test_num_samples = [train_samples,
                                      eval_iters * args.global_batch_size,
                                      test_iters * args.global_batch_size]
        print_rank_0(' > datasets target sizes (minimum size):')
        print_rank_0('    train:      {}'.format(train_val_test_num_samples[0]))
        print_rank_0('    validation: {}'.format(train_val_test_num_samples[1]))
        print_rank_0('    test:       {}'.format(train_val_test_num_samples[2]))

        # Build the datasets.
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets_provider(
            train_val_test_num_samples)

        # Build dataloders.
        train_dataloader = build_pretraining_data_loader(
            train_ds, args.consumed_train_samples)
        valid_dataloader = build_pretraining_data_loader(
            valid_ds, args.consumed_valid_samples)
        test_dataloader = build_pretraining_data_loader(test_ds, 0)

        # Flags to know if we need to do training/validation/testing.
        do_train = train_dataloader is not None and args.train_iters > 0
        do_valid = valid_dataloader is not None and args.eval_iters > 0
        do_test = test_dataloader is not None and args.eval_iters > 0
        # Need to broadcast num_tokens and num_type_tokens.
        flags = torch.cuda.LongTensor(
            [int(do_train), int(do_valid), int(do_test)])
    else:
        flags = torch.cuda.LongTensor([0, 0, 0])

    # Broadcast num tokens.
    torch.distributed.broadcast(flags,
                                mpu.get_tensor_model_parallel_src_rank(),
                                group=mpu.get_tensor_model_parallel_group())
    args.do_train = flags[0].item()
    args.do_valid = flags[1].item()
    args.do_test = flags[2].item()

    # Build iterators.
    if train_dataloader is not None:
        train_data_iterator = iter(train_dataloader)
    else:
        train_data_iterator = None

    if valid_dataloader is not None:
        valid_data_iterator = iter(valid_dataloader)
    else:
        valid_data_iterator = None

    if test_dataloader is not None:
        test_data_iterator = iter(test_dataloader)
    else:
        test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_data_iterator
