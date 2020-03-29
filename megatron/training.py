# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

"""Pretrain utilities"""

from datetime import datetime
import math

import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from apex.optimizers import FusedAdam as Adam

from megatron.global_vars import get_args
from megatron.global_vars import get_timers
from megatron.global_vars import get_tensorboard_writer
from megatron.global_vars import get_adlr_autoresume
from megatron.initialize import initialize_megatron

from megatron import mpu
from megatron.fp16 import FP16_Module
from megatron.fp16 import FP16_Optimizer
from megatron.learning_rates import AnnealingLR
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import get_params_for_weight_decay_optimization
from megatron.utils import check_adlr_autoresume_termination
from megatron.checkpointing import load_checkpoint
from megatron import print_rank_0
from megatron.utils import report_memory
from megatron.checkpointing import save_checkpoint


def run(top_level_message, train_val_test_data_provider,
        model_provider, forward_step_func, extra_args_provider=None,
        args_defaults={}):
    """Main training program.

    This function will run the followings in the order provided:
        1) get input arguments.
        2) initialize distributed and seeds.
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) setup model, optimizer and lr schedule using the model_provider.
        5) train the modle using the forward_step_func.

    Arguments:
        top_level_message: a meesage to print at the top of the run.
        train_val_test_data_provider: a function that takes `args` as input
            and returns `train, val, test` dataloaders. Note that args are
            passed and can be modified in case we need to use some parameters
            later. For example, we can set vocab size using
                args.vocab_size = ...
            and later use this value in `model_provider`.
        model_provider: a function that takes `args` and returns a vanilla
            version of the model. By vanilla we mean a simple model on cpu
            with no fp16 or ddp.
        forward_step_func: a function that takes a `data iterator`, `model`,
            `args`, and `timers` and returns a `loss` scalar with a dictionary
            with key:values being the info we would like to monitor during
            training, for example `lm-loss: value`. We also require that this
            function add `batch generator` to the timers class.
    """

    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults)
    args = get_args()
    timers = get_timers()
    writer = get_tensorboard_writer()

    # Data stuff.
    train_data, val_data, test_data = train_val_test_data_provider(args)

    # Model, optimizer, and learning rate.
    model, optimizer, lr_scheduler = setup_model_and_optimizer(model_provider,
                                                               args)

    # Train, validation, and test data.
    train_data_iterator, val_data_iterator, \
        test_data_iterator = get_train_val_test_data_iterators(train_data,
                                                               val_data,
                                                               test_data,
                                                               args)

    iteration = 0
    if args.train_iters > 0:
        if args.do_train:
            iteration, _ = train(forward_step_func, model,
                                 optimizer, lr_scheduler,
                                 train_data_iterator, val_data_iterator,
                                 timers, args, writer)

    if args.do_valid:
        prefix = 'the end of training for val data'
        evaluate_and_print_results(prefix, forward_step_func,
                                   val_data_iterator, model,
                                   args, writer, iteration,
                                   timers, False)

    if args.save and iteration != 0:
        save_checkpoint(iteration, model, optimizer, lr_scheduler)

    if args.do_test:
        # Run on test data.
        prefix = 'the end of training for test data'
        evaluate_and_print_results(prefix, forward_step_func,
                                   test_data_iterator, model,
                                   args, None, 0, timers, True)


def get_model(model_provider_func, args):
    """Build the model."""

    # Build model on cpu.
    model = model_provider_func(args)

    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank(),
            sum([p.nelement() for p in model.parameters()])), flush=True)

    # GPU allocation.
    model.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16:
        model = FP16_Module(model)

    # Wrap model for distributed training."""
    if args.DDP_impl == 'torch':
        i = torch.cuda.current_device()
        args.DDP_type = torchDDP
        model = args.DDP_type(model, device_ids=[i], output_device=i,
                              process_group=mpu.get_data_parallel_group())
        return model
    if args.DDP_impl == 'local':
        args.DDP_type = LocalDDP
        model = args.DDP_type(model)
        return model

    print_rank_0('Unknown DDP implementation specified: {}. '
                 'Exiting.'.format(args.DDP_impl))
    exit()
    return model


def get_optimizer(model, args):
    """Set up the optimizer."""

    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, (args.DDP_type, FP16_Module)):
        model = model.module
    param_groups = get_params_for_weight_decay_optimization(model)

    # Add model parallel attribute if it is not set.
    for param_group in param_groups:
        for param in param_group['params']:
            if not hasattr(param, 'model_parallel'):
                param.model_parallel = False

    # Use Adam.
    optimizer = Adam(param_groups,
                     lr=args.lr, weight_decay=args.weight_decay)

    # Wrap into fp16 optimizer.
    if args.fp16:
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=args.loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale,
                                   dynamic_loss_args={
                                       'scale_window': args.loss_scale_window,
                                       'min_scale':args.min_scale,
                                       'delayed_shift': args.hysteresis})

    return optimizer


def get_learning_rate_scheduler(optimizer, args):
    """Build the learning rate scheduler."""

    # Add linear learning rate scheduler.
    if args.lr_decay_iters is not None:
        num_iters = args.lr_decay_iters
    else:
        num_iters = args.train_iters
    num_iters = max(1, num_iters)
    init_step = -1
    warmup_iter = args.warmup * num_iters
    lr_scheduler = AnnealingLR(
        optimizer,
        start_lr=args.lr,
        warmup_iter=warmup_iter,
        num_iters=num_iters,
        decay_style=args.lr_decay_style,
        last_iter=init_step,
        min_lr=args.min_lr,
        use_checkpoint_lr_scheduler=args.use_checkpoint_lr_scheduler,
        override_lr_scheduler=args.override_lr_scheduler)

    return lr_scheduler


def setup_model_and_optimizer(model_provider_func, args):
    """Setup model and optimizer."""

    model = get_model(model_provider_func, args)
    optimizer = get_optimizer(model, args)
    lr_scheduler = get_learning_rate_scheduler(optimizer, args)

    if args.load is not None:
        args.iteration = load_checkpoint(model, optimizer, lr_scheduler)
    else:
        args.iteration = 0

    return model, optimizer, lr_scheduler


def backward_step(optimizer, model, loss, args, timers):
    """Backward step."""

    # Backward pass.
    optimizer.zero_grad()
    if args.fp16:
        optimizer.backward(loss, update_master_grads=False)
    else:
        loss.backward()

    # All-reduce if needed.
    if args.DDP_impl == 'local':
        timers('allreduce').start()
        model.allreduce_params(reduce_after=False,
                               fp32_allreduce=args.fp32_allreduce)
        timers('allreduce').stop()

    # Update master gradients.
    if args.fp16:
        optimizer.update_master_grads()

    # Clipping gradients helps prevent the exploding gradient.
    if args.clip_grad > 0:
        if not args.fp16:
            mpu.clip_grad_norm(model.parameters(), args.clip_grad)
        else:
            optimizer.clip_master_grads(args.clip_grad)


def train_step(forward_step_func, data_iterator, model, optimizer, lr_scheduler,
               args, timers):
    """Single training step."""

    # Forward model for one step.
    timers('forward').start()
    loss, loss_reduced = forward_step_func(data_iterator, model, args, timers)
    timers('forward').stop()

    # Calculate gradients, reduce across processes, and clip.
    timers('backward').start()
    backward_step(optimizer, model, loss, args, timers)
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

    return loss_reduced, skipped_iter


def training_log(loss_dict, total_loss_dict, learning_rate, iteration,
                 loss_scale, report_memory_flag, writer, args, timers):

    # Update losses.
    for key in loss_dict:
        total_loss_dict[key] = total_loss_dict.get(key, 0.) + loss_dict[key]

    # Logging.
    timers_to_log = []
    def add_to_logging(name):
        if name in timers.timers:
            timers_to_log.append(name)
    add_to_logging('forward')
    add_to_logging('backward')
    add_to_logging('allreduce')
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
        for key in total_loss_dict:
            avg = total_loss_dict[key].item() / args.log_interval
            log_string += ' {}: {:.6E} |'.format(key, avg)
            total_loss_dict[key] = 0.0
        if args.fp16:
            log_string += ' loss scale: {:.1f} |'.format(loss_scale)
        print_rank_0(log_string)
        if report_memory_flag:
            report_memory('after {} iterations'.format(iteration))
            report_memory_flag = False
        timers.log(timers_to_log, normalizer=args.log_interval)

    return report_memory_flag


def train(forward_step_func, model, optimizer, lr_scheduler,
          train_data_iterator, val_data_iterator, timers, args, writer):
    """Train the model function."""

    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    total_loss_dict = {}

    # Iterations.
    iteration = args.iteration
    skipped_iters = 0

    timers('interval time').start()
    report_memory_flag = True
    while iteration < args.train_iters:
        loss_dict, skipped_iter = train_step(forward_step_func,
                                             train_data_iterator,
                                             model,
                                             optimizer,
                                             lr_scheduler,
                                             args, timers)
        skipped_iters += skipped_iter
        iteration += 1

        # Logging.
        report_memory_flag = training_log(loss_dict, total_loss_dict,
                                          optimizer.param_groups[0]['lr'],
                                          iteration, optimizer.loss_scale,
                                          report_memory_flag, writer, args,
                                          timers)

        # Autoresume
        if (iteration % args.adlr_autoresume_interval == 0) and \
           args.adlr_autoresume:
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
                                       val_data_iterator, model, args,
                                       writer, iteration, timers, False)

        if args.exit_interval and iteration % args.exit_interval == 0:
            torch.distributed.barrier()
            time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            rank = torch.distributed.get_rank()
            print('rank: {} | time: {} | exiting the program at iteration {}'.
                  format(rank, time_str, iteration), flush=True)
            exit()

    return iteration, skipped_iters


def evaluate(forward_step_func, data_iterator, model,
             args, timers, verbose=False):
    """Evaluation."""

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
            # Forward evaluation.
            _, loss_dict = forward_step_func(data_iterator, model,
                                             args, timers)
            # Reduce across processes.
            for key in loss_dict:
                total_loss_dict[key] = total_loss_dict.get(key, 0.) + \
                                       loss_dict[key]
    # Move model back to the train mode.
    model.train()

    for key in total_loss_dict:
        total_loss_dict[key] /= args.eval_iters

    return total_loss_dict


def evaluate_and_print_results(prefix, forward_step_func,
                               data_iterator, model,
                               args, writer, iteration,
                               timers, verbose=False):
    """Helper function to evaluate and dump results on screen."""
    total_loss_dict = evaluate(forward_step_func, data_iterator, model,
                               args, timers, verbose)
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
    print_rank_0('-' * length)
    print_rank_0(string)
    print_rank_0('-' * length)


def get_train_val_test_data_iterators(train_data, val_data, test_data, args):
    """Build train/validation/test iterators"""

    # Shift the start iterations.
    if train_data is not None:
        train_data.batch_sampler.start_iter = args.iteration % \
                                              len(train_data)
        print_rank_0('setting training data start iteration to {}'.
                     format(train_data.batch_sampler.start_iter))
    if val_data is not None:
        start_iter_val = (args.iteration // args.eval_interval) * \
                         args.eval_iters
        val_data.batch_sampler.start_iter = start_iter_val % \
                                            len(val_data)
        print_rank_0('setting validation data start iteration to {}'.
                     format(val_data.batch_sampler.start_iter))

    if train_data is not None:
        train_data_iterator = iter(train_data)
    else:
        train_data_iterator = None

    if val_data is not None:
        val_data_iterator = iter(val_data)
    else:
        val_data_iterator = None

    if test_data is not None:
        test_data_iterator = iter(test_data)
    else:
        test_data_iterator = None

    return train_data_iterator, val_data_iterator, test_data_iterator
