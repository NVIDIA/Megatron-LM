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
import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from apex.optimizers import FusedAdam as Adam

from megatron import get_args
from megatron import get_timers
from megatron import get_tensorboard_writer
from megatron import mpu
from megatron import print_rank_0
from megatron import print_rank_last
from megatron.checkpointing import load_checkpoint
from megatron.checkpointing import save_checkpoint
from megatron.fp16 import FP16_Module
from megatron.fp16 import FP16_Optimizer
from megatron.initialize import initialize_megatron
from megatron.learning_rates import AnnealingLR
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import get_params_for_weight_decay_optimization
from megatron.model.realm_model import ICTBertModel
from megatron.utils import check_adlr_autoresume_termination
from megatron.utils import make_data_loader
from megatron.utils import report_memory


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

    args = get_args()
    timers = get_timers()

    # Model, optimizer, and learning rate.
    timers('model and optimizer').start()
    model, optimizer, lr_scheduler = setup_model_and_optimizer(model_provider)
    timers('model and optimizer').stop()

    # Data stuff.
    timers('train/valid/test data iterators').start()
    train_data_iterator, valid_data_iterator, test_data_iterator \
        = build_train_valid_test_data_iterators(
            train_valid_test_dataset_provider)
    timers('train/valid/test data iterators').stop()

    # Print setup timing.
    print_rank_0('done with setups ...')
    timers.log(['model and optimizer', 'train/valid/test data iterators'])
    print_rank_0('training ...')

    iteration = 0
    if args.do_train and args.train_iters > 0:
        iteration = train(forward_step_func,
                          model, optimizer, lr_scheduler,
                          train_data_iterator, valid_data_iterator)

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


def get_model(model_provider_func):
    """Build the model."""
    args = get_args()

    # Build model on cpu.
    model = model_provider_func()

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
        model = FP16_Module(model)

    # Wrap model for distributed training."""
    if args.num_microbatches_in_minibatch > 1:
        assert args.DDP_impl == 'local'

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


def get_optimizer(model):
    """Set up the optimizer."""
    args = get_args()

    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, (torchDDP, LocalDDP, FP16_Module)):
        model = model.module
    param_groups = get_params_for_weight_decay_optimization(model)

    # Add model parallel attribute if it is not set.
    for param_group in param_groups:
        for param in param_group['params']:
            if not hasattr(param, 'tensor_model_parallel'):
                param.tensor_model_parallel = False

    # Use Adam.
    optimizer = Adam(param_groups, lr=args.lr, weight_decay=args.weight_decay,
        betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_eps)

    # Wrap into fp16 optimizer.
    if args.fp16:
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=args.loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale,
                                   dynamic_loss_args={
                                       'scale_window': args.loss_scale_window,
                                       'min_scale': args.min_scale,
                                       'delayed_shift': args.hysteresis})

    return optimizer


def get_learning_rate_scheduler(optimizer):
    """Build the learning rate scheduler."""
    args = get_args()

    # Add linear learning rate scheduler.
    if args.lr_decay_iters is not None:
        num_iters = args.lr_decay_iters
    else:
        num_iters = args.train_iters
    num_iters = max(1, num_iters)
    init_step = 0
    warmup_iter = args.warmup * num_iters
    lr_scheduler = AnnealingLR(
        optimizer,
        start_lr=args.lr,
        warmup_iter=warmup_iter,
        total_iters=num_iters,
        decay_style=args.lr_decay_style,
        last_iter=init_step,
        min_lr=args.min_lr,
        use_checkpoint_lr_scheduler=args.use_checkpoint_lr_scheduler,
        override_lr_scheduler=args.override_lr_scheduler)

    return lr_scheduler


def setup_model_and_optimizer(model_provider_func):
    """Setup model and optimizer."""
    args = get_args()

    model = get_model(model_provider_func)
    optimizer = get_optimizer(model)
    lr_scheduler = get_learning_rate_scheduler(optimizer)

    if args.load is not None:
        args.iteration = load_checkpoint(model, optimizer, lr_scheduler)
    else:
        args.iteration = 0

    # get model without FP16 and/or TorchDDP wrappers
    unwrapped_model = model
    while hasattr(unwrapped_model, 'module'):
        unwrapped_model = unwrapped_model.module

    if args.iteration == 0 and hasattr(unwrapped_model, 'init_state_dict_from_bert'):
        print("Initializing ICT from pretrained BERT model", flush=True)
        unwrapped_model.init_state_dict_from_bert()

    return model, optimizer, lr_scheduler


def communicate(tensor_send_next, tensor_send_prev, recv_forward, recv_backward):
    """Communicate tensors between stages using torch.distributed.ring_exchange(.) API."""
    args = get_args()

    # Create placeholder tensors for receive in forward and backward directions
    # if needed.
    tensor_recv_prev = None
    tensor_recv_next = None
    tensor_shape = (args.batch_size, args.seq_length, args.hidden_size)
    if recv_forward:
        tensor_recv_prev = torch.empty(tensor_shape,
                                       requires_grad=True,
                                       dtype=args.params_dtype).cuda()
    if recv_backward:
        tensor_recv_next = torch.empty(tensor_shape,
                                       requires_grad=True,
                                       dtype=args.params_dtype).cuda()

    # Send tensors in both the forward and backward directions as appropriate.
    torch.distributed.ring_exchange(tensor_send_prev=tensor_send_prev,
                                    tensor_recv_prev=tensor_recv_prev,
                                    tensor_send_next=tensor_send_next,
                                    tensor_recv_next=tensor_recv_next,
                                    group=mpu.get_pipeline_model_parallel_group())

    return tensor_recv_prev, tensor_recv_next


def backward_step(optimizer, model, input_tensor, output_tensor, output_tensor_grad):
    """Backward step."""
    args = get_args()
    timers = get_timers()

    # Retain the grad on the input_tensor.
    if input_tensor is not None:
        input_tensor.retain_grad()

    # Backward pass.
    timers('backward-backward').start()
    if args.fp16:
        optimizer.backward(output_tensor, update_master_grads=False,
                           output_tensor_grad=output_tensor_grad)
    else:
        torch.autograd.backward(output_tensor, grad_tensors=output_tensor_grad)
    timers('backward-backward').stop()

    # Collect the grad of the input_tensor.
    input_tensor_grad = None
    if input_tensor is not None:
        input_tensor_grad = input_tensor.grad

    return input_tensor_grad


def forward_step_with_communication(forward_step_func, data_iterator, model,
                                    input_tensors, output_tensors,
                                    losses_reduced, timers):
    if not mpu.is_pipeline_first_stage():
        input_tensor, _ = communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_forward=True,
            recv_backward=False)
    else:
        input_tensor = None

    # Forward model for one step.
    output_tensor = forward_step_func(data_iterator, model, input_tensor)

    if mpu.is_pipeline_last_stage():
        loss, loss_reduced = output_tensor
        output_tensor = loss
        losses_reduced.append(loss_reduced)
    else:
        communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_forward=False,
            recv_backward=False)

    input_tensors.append(input_tensor)
    output_tensors.append(output_tensor)


def backward_step_with_communication(optimizer, model, input_tensors, output_tensors, timers):
    """Backward step."""
    input_tensor = input_tensors.pop(0)
    output_tensor = output_tensors.pop(0)

    if mpu.is_pipeline_last_stage():
        output_tensor_grad = None
    else:
        _, output_tensor_grad = communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_forward=False,
            recv_backward=True)

    # Backward pass for one step.
    input_grad_tensor = \
        backward_step(optimizer, model, input_tensor, output_tensor, output_tensor_grad)

    if not mpu.is_pipeline_first_stage():
        communicate(
            tensor_send_next=None,
            tensor_send_prev=input_grad_tensor,
            recv_forward=False,
            recv_backward=False)


def train_step(forward_step_func, data_iterator,
               model, optimizer, lr_scheduler):
    """Single training step."""
    args = get_args()
    timers = get_timers()

    # Set grad to zero.
    if args.fp16:
        optimizer.zero_grad(set_grads_to_None=True)
    else:
        optimizer.zero_grad()

    # Compute number of microbatches in a minibatch.
    num_microbatches_in_minibatch = args.num_microbatches_in_minibatch
    # TODO: Switch to the following schedule to facilitate more
    # memory-efficient training.
    # num_warmup_microbatches = \
    #     (torch.distributed.get_world_size(group=mpu.get_pipeline_model_parallel_group()) -
    #      torch.distributed.get_rank(group=mpu.get_pipeline_model_parallel_group()) - 1)
    # num_warmup_microbatches = min(
    #     num_warmup_microbatches,
    #     num_microbatches_in_minibatch)
    # For now, perform training without warmup. Perform forward
    # passes for all microbatches, then backward passes for all
    # microbatches.
    num_warmup_microbatches = num_microbatches_in_minibatch

    input_tensors = []
    output_tensors = []
    losses_reduced = []

    # Run warmup forward passes.
    timers('forward').start()
    for i in range(num_warmup_microbatches):
        forward_step_with_communication(
            forward_step_func, data_iterator, model,
            input_tensors, output_tensors,
            losses_reduced, timers)
    timers('forward').stop()

    # Run cooldown backward passes.
    timers('backward').start()
    for i in range(num_warmup_microbatches):
        backward_step_with_communication(
            optimizer, model, input_tensors, output_tensors, timers)

    # All-reduce if needed.
    if args.DDP_impl == 'local':
        timers('allreduce').start()
        model.allreduce_params(reduce_after=False,
                               fp32_allreduce=args.fp32_allreduce)
        timers('allreduce').stop()

    # Update master gradients.
    timers('backward-master-grad').start()
    if args.fp16:
        optimizer.update_master_grads()
    timers('backward-master-grad').stop()

    # All-reduce across first and last stages.
    timers('backward-embedding-all-reduce').start()
    if (mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()) and \
            args.pipeline_model_parallel_size > 1:
        unwrapped_model = model
        while isinstance(unwrapped_model, (torchDDP, LocalDDP, FP16_Module)):
            unwrapped_model = unwrapped_model.module

        word_embeddings_weight = unwrapped_model.word_embeddings_weight()
        torch.distributed.all_reduce(word_embeddings_weight.grad,
                                     group=mpu.get_embedding_group())
    timers('backward-embedding-all-reduce').stop()

    # Clipping gradients helps prevent the exploding gradient.
    timers('backward-clip-grad').start()
    if args.clip_grad > 0.:
        if not args.fp16:
            named_parameters = model.named_parameters()
            parameters = []
            parameter_names = []
            for parameter_name, parameter in model.named_parameters():
                parameters.append(parameter)
                parameter_names.append(parameter_name)
            mpu.clip_grad_norm(parameters, args.clip_grad,
                               parameter_names=parameter_names)
        else:
            optimizer.clip_master_grads(args.clip_grad)
    timers('backward-clip-grad').stop()
    timers('backward').stop()

    # Update parameters.
    timers('optimizer').start()
    optimizer.step()
    timers('optimizer').stop()

    # Update learning rate.
    skipped_iter = 0
    if not (args.fp16 and optimizer.overflow):
        lr_scheduler.step()
    else:
        skipped_iter = 1

    if mpu.is_pipeline_last_stage():
        # Average loss across microbatches.
        loss_reduced = {}
        for key in losses_reduced[0]:
            losses_reduced_for_key = [x[key] for x in losses_reduced]
            loss_reduced[key] = sum(losses_reduced_for_key) / \
                    len(losses_reduced_for_key)
        return loss_reduced, skipped_iter
    return {}, skipped_iter


def training_log(loss_dict, total_loss_dict, learning_rate, iteration,
                 loss_scale, report_memory_flag, skipped_iter):
    """Log training information such as losses, timing, ...."""
    args = get_args()
    timers = get_timers()
    writer = get_tensorboard_writer()

    # Update losses.
    skipped_iters_key = 'skipped iterations'
    total_loss_dict[skipped_iters_key] = total_loss_dict.get(
        skipped_iters_key, 0) + skipped_iter
    got_nan_key = 'got nan'

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

    total_loss_dict[got_nan_key] = total_loss_dict.get(
        got_nan_key, 0) + int(got_nan)

    # Logging.
    timers_to_log = []

    def add_to_logging(name):
        if name in timers.timers:
            timers_to_log.append(name)
    add_to_logging('forward')
    add_to_logging('backward')
    add_to_logging('backward-backward')
    add_to_logging('backward-allreduce')
    add_to_logging('backward-master-grad')
    add_to_logging('backward-embedding-all-reduce')
    add_to_logging('backward-clip-grad')
    add_to_logging('optimizer')
    add_to_logging('batch generator')

    # Tensorboard values.
    if writer and torch.distributed.get_rank() == 0:
        writer.add_scalar('learning_rate', learning_rate, iteration)
        for key in loss_dict:
            writer.add_scalar(key, loss_dict[key], iteration)
        if args.fp16:
            writer.add_scalar('loss_scale', loss_scale, iteration)
        normalizer = iteration % args.log_interval
        if normalizer == 0:
            normalizer = args.log_interval
        timers.write(timers_to_log, writer, iteration,
                     normalizer=normalizer)

    if iteration % args.log_interval == 0:
        elapsed_time = timers('interval time').elapsed()
        if writer and torch.distributed.get_rank() == 0:
            writer.add_scalar('iteration_time',
                              elapsed_time / args.log_interval, iteration)
        log_string = ' iteration {:8d}/{:8d} |'.format(iteration,
                                                       args.train_iters)
        log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
            elapsed_time * 1000.0 / args.log_interval)
        log_string += ' learning rate: {:.3E} |'.format(learning_rate)
        num_iterations = max(
            1, args.log_interval - total_loss_dict[skipped_iters_key])
        for key in total_loss_dict:
            if key not in [skipped_iters_key, got_nan_key]:
                avg = total_loss_dict[key].item() / float(num_iterations)
                if avg > 0.0:
                    log_string += ' {}: {:.6E} |'.format(key, avg)
                total_loss_dict[key] = torch.cuda.FloatTensor([0.0])
        if args.fp16:
            log_string += ' loss scale: {:.1f} |'.format(loss_scale)
        log_string += ' number of skipped iterations: {:3d} |'.format(
            total_loss_dict[skipped_iters_key])
        log_string += ' number of nan iterations: {:3d} |'.format(
            total_loss_dict[got_nan_key])
        total_loss_dict[skipped_iters_key] = 0
        total_loss_dict[got_nan_key] = 0
        print_rank_last(log_string)
        if report_memory_flag:
            report_memory('after {} iterations'.format(iteration))
            report_memory_flag = False
        timers.log(timers_to_log, normalizer=args.log_interval)

    return report_memory_flag


def train(forward_step_func, model, optimizer, lr_scheduler,
          train_data_iterator, valid_data_iterator):
    """Train the model function."""
    args = get_args()
    timers = get_timers()

    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    total_loss_dict = {}

    # Iterations.
    iteration = args.iteration

    timers('interval time').start()
    report_memory_flag = True
    while iteration < args.train_iters:
        loss_dict, skipped_iter = train_step(forward_step_func,
                                             train_data_iterator,
                                             model,
                                             optimizer,
                                             lr_scheduler)
        iteration += 1

        # Logging.
        loss_scale = None
        if args.fp16:
            loss_scale = optimizer.loss_scale
        report_memory_flag = training_log(loss_dict, total_loss_dict,
                                          optimizer.param_groups[0]['lr'],
                                          iteration, loss_scale,
                                          report_memory_flag, skipped_iter)

        # Autoresume
        if args.adlr_autoresume and \
           (iteration % args.adlr_autoresume_interval == 0):
            check_adlr_autoresume_termination(iteration, model, optimizer,
                                              lr_scheduler)

        # Checkpointing
        if args.save and args.save_interval and \
           iteration % args.save_interval == 0:
            save_checkpoint(iteration, model, optimizer, lr_scheduler)

        # Evaluation
        if args.eval_interval and iteration % args.eval_interval == 0 and \
           args.do_valid:
            prefix = 'iteration {}'.format(iteration)
            evaluate_and_print_results(prefix, forward_step_func,
                                       valid_data_iterator, model,
                                       iteration, False)

        if args.exit_interval and iteration % args.exit_interval == 0:
            torch.distributed.barrier()
            time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            rank = torch.distributed.get_rank()
            print_rank_0('rank: {} | time: {} | exiting the program at '
                         'iteration {}'.format(rank, time_str, iteration))
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
                    total_loss_dict[key] = total_loss_dict.get(key, 0.) + \
                        loss_dict[key]
            else:
                communicate(
                    tensor_send_next=output_tensor,
                    tensor_send_prev=None,
                    recv_forward=False,
                    recv_backward=False)

    # Move model back to the train mode.
    model.train()

    for key in total_loss_dict:
        total_loss_dict[key] /= args.eval_iters

    return total_loss_dict


def evaluate_and_print_results(prefix, forward_step_func,
                               data_iterator, model,
                               iteration, verbose=False):
    """Helper function to evaluate and dump results on screen."""
    writer = get_tensorboard_writer()

    total_loss_dict = evaluate(forward_step_func, data_iterator, model, verbose)
    string = ' validation loss at {} | '.format(prefix)
    for key in total_loss_dict:
        string += '{} value: {:.6E} | '.format(key, total_loss_dict[key].item())
        ppl = math.exp(min(20, total_loss_dict[key].item()))
        string += '{} PPL: {:.6E} | '.format(key, ppl)
        if writer and torch.distributed.get_rank() == 0:
            writer.add_scalar('{} value'.format(key),
                              total_loss_dict[key].item(),
                              iteration)
            writer.add_scalar('{} ppl'.format(key), ppl, iteration)

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
    # Data loader only on rank 0 of each model parallel group.
    if mpu.get_tensor_model_parallel_rank() == 0:
        # Rank, size, and global batch size.
        data_parallel_size = mpu.get_data_parallel_world_size()
        global_batch_size = args.batch_size * data_parallel_size

        # Number of train/valid/test samples.
        train_iters = args.train_iters
        eval_iters = (train_iters // args.eval_interval + 1) * args.eval_iters
        test_iters = args.eval_iters
        train_val_test_num_samples = [train_iters * global_batch_size,
                                      eval_iters * global_batch_size,
                                      test_iters * global_batch_size]
        print_rank_0(' > datasets target sizes (minimum size):')
        print_rank_0('    train:      {}'.format(train_val_test_num_samples[0]))
        print_rank_0('    validation: {}'.format(train_val_test_num_samples[1]))
        print_rank_0('    test:       {}'.format(train_val_test_num_samples[2]))

        # Build the datasets.
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets_provider(
            train_val_test_num_samples)

        # Build dataloders.
        train_dataloader = make_data_loader(train_ds)
        valid_dataloader = make_data_loader(valid_ds)
        test_dataloader = make_data_loader(test_ds)

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

    # Shift the start iterations.
    if train_dataloader is not None:
        train_dataloader.batch_sampler.start_iter = args.iteration % \
            len(train_dataloader)
        print_rank_0('setting training data start iteration to {}'.
                     format(train_dataloader.batch_sampler.start_iter))
    if valid_dataloader is not None:
        start_iter_val = (args.iteration // args.eval_interval) * \
            args.eval_iters
        valid_dataloader.batch_sampler.start_iter = start_iter_val % \
            len(valid_dataloader)
        print_rank_0('setting validation data start iteration to {}'.
                     format(valid_dataloader.batch_sampler.start_iter))

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
