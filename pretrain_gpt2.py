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

"""Pretrain GPT2"""

from datetime import datetime
import os
import random
import math
import numpy as np
import torch

from arguments import get_args
from configure_data import configure_data
from megatron.fp16 import FP16_Module
from megatron.fp16 import FP16_Optimizer
from megatron.learning_rates import AnnealingLR
from megatron.model import GPT2Model
from megatron.model import gpt2_get_params_for_weight_decay_optimization
from megatron import mpu
from apex.optimizers import FusedAdam as Adam
from megatron.utils import Timers
from megatron.utils import save_checkpoint
from megatron.utils import load_checkpoint
from megatron.utils import report_memory
from megatron.utils import print_args
from megatron.utils import print_params_min_max_norm
from megatron.utils import print_rank_0
from megatron.utils import enable_adlr_autoresume
from megatron.utils import check_adlr_autoresume_termination
from megatron.utils import initialize_distributed
from megatron.utils import set_random_seed
from megatron.utils import wrap_model_for_distributed_training

from gpt2_data_loader import make_gpt2_dataloaders

def get_model(args):
    """Build the model."""

    print_rank_0('building GPT2 model ...')
    model = GPT2Model(num_layers=args.num_layers,
                      vocab_size=args.vocab_size,
                      hidden_size=args.hidden_size,
                      num_attention_heads=args.num_attention_heads,
                      embedding_dropout_prob=args.hidden_dropout,
                      attention_dropout_prob=args.attention_dropout,
                      output_dropout_prob=args.hidden_dropout,
                      max_sequence_length=args.max_position_embeddings,
                      checkpoint_activations=args.checkpoint_activations,
                      checkpoint_num_layers=args.checkpoint_num_layers,
                      parallel_output=True)

    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank(),
            sum([p.nelement() for p in model.parameters()])), flush=True)

    # GPU allocation.
    model.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16:
        model = FP16_Module(model)

    # Wrap model for distributed training.
    model = wrap_model_for_distributed_training(model, args)

    return model


def get_optimizer(model, args):
    """Set up the optimizer."""

    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, (args.DDP_type, FP16_Module)):
        model = model.module
    param_groups = gpt2_get_params_for_weight_decay_optimization(model)

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
    lr_scheduler = AnnealingLR(optimizer,
                               start_lr=args.lr,
                               warmup_iter=warmup_iter,
                               num_iters=num_iters,
                               decay_style=args.lr_decay_style,
                               last_iter=init_step,
                               min_lr=args.min_lr,
                               use_checkpoint_lr_scheduler=args.use_checkpoint_lr_scheduler,
                               override_lr_scheduler=args.override_lr_scheduler)

    return lr_scheduler


def setup_model_and_optimizer(args):
    """Setup model and optimizer."""

    model = get_model(args)
    optimizer = get_optimizer(model, args)
    lr_scheduler = get_learning_rate_scheduler(optimizer, args)

    if args.load is not None:
        args.iteration = load_checkpoint(model, optimizer, lr_scheduler, args)
    else:
        args.iteration = 0

    return model, optimizer, lr_scheduler


def get_masks_and_position_ids(data,
                               eod_token,
                               reset_position_ids,
                               reset_attention_mask,
                               eod_mask_loss):

    # Extract batch size and sequence length.
    batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(torch.ones(
        (att_mask_batch, seq_length, seq_length), device=data.device)).view(
            att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i+1):, :(i+1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i+1):] -= (i + 1 - prev_index)
                    prev_index = i + 1

    return attention_mask, loss_mask, position_ids


def get_batch(data_iterator, args, timers):
    ''' get_batch subdivides the source data into chunks of
    length args.seq_length. If source is equal to the example
    output of the data loading example, with a seq_length limit
    of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the data loader. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM. A Variable representing an appropriate
    shard reset mask of the same dimensions is also returned.
    '''
    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    timers('data loader').start()
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    timers('data loader').stop()
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_masks_and_position_ids(
        tokens,
        args.eod_token,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)
    # Convert
    if args.fp16:
        attention_mask = attention_mask.half()

    return tokens, labels, loss_mask, attention_mask, position_ids


def forward_step(data_iterator, model, args, timers):
    """Forward step."""

    # Get the batch.
    timers('batch generator').start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator, args, timers)
    timers('batch generator').stop()

    # Forward model.
    output = model(tokens, position_ids, attention_mask)
    losses = mpu.vocab_parallel_cross_entropy(output.contiguous().float(),
                                              labels)
    loss_mask = loss_mask.view(-1)
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    return loss


def backward_step(optimizer, model, lm_loss, args, timers):
    """Backward step."""

    # Total loss.
    loss = lm_loss

    # Backward pass.
    optimizer.zero_grad()
    if args.fp16:
        optimizer.backward(loss, update_master_grads=False)
    else:
        loss.backward()

    # Reduce across processes.
    lm_loss_reduced = lm_loss

    reduced_losses = lm_loss.view(1)
    torch.distributed.all_reduce(reduced_losses.data)
    reduced_losses.data = reduced_losses.data / args.world_size
    if args.DDP_impl == 'local':
        timers('allreduce').start()
        model.allreduce_params(reduce_after=False,
                               fp32_allreduce=args.fp32_allreduce)
        timers('allreduce').stop()
    lm_loss_reduced = reduced_losses

    # Update master gradients.
    if args.fp16:
        optimizer.update_master_grads()

    # Clipping gradients helps prevent the exploding gradient.
    if args.clip_grad > 0:
        if not args.fp16:
            mpu.clip_grad_norm(model.parameters(), args.clip_grad)
        else:
            optimizer.clip_master_grads(args.clip_grad)

    return lm_loss_reduced


def train_step(data_iterator, model, optimizer, lr_scheduler,
               args, timers):
    """Single training step."""

    # Forward model for one step.
    timers('forward').start()
    lm_loss = forward_step(data_iterator, model, args, timers)
    timers('forward').stop()

    # Calculate gradients, reduce across processes, and clip.
    timers('backward').start()
    lm_loss_reduced = backward_step(optimizer, model, lm_loss, args, timers)
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

    return lm_loss_reduced, skipped_iter


def train(model, optimizer, lr_scheduler,
          train_data_iterator, val_data_iterator, timers, args, writer):
    """Train the model."""

    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    total_lm_loss = 0.0

    # Iterations.
    iteration = args.iteration
    skipped_iters = 0

    timers('interval time').start()
    report_memory_flag = True
    while iteration < args.train_iters:

        lm_loss, skipped_iter = train_step(train_data_iterator,
                                           model,
                                           optimizer,
                                           lr_scheduler,
                                           args, timers)
        skipped_iters += skipped_iter
        iteration += 1

        # Update losses.
        current_lm_loss = lm_loss.data.detach().float()
        total_lm_loss += current_lm_loss

        # Logging.

        if args.DDP_impl == 'torch':
            timers_to_log = ['forward', 'backward', 'optimizer',
                            'batch generator', 'data loader']
        else:
            timers_to_log = ['forward', 'backward', 'allreduce', 'optimizer',
                             'batch generator', 'data loader']

        learning_rate = optimizer.param_groups[0]['lr']

        if writer and args.rank == 0:
            writer.add_scalar('learning_rate', learning_rate, iteration)
            writer.add_scalar('train_loss', current_lm_loss, iteration)
            if args.fp16:
                writer.add_scalar('loss_scale', optimizer.loss_scale, iteration)
            normalizer = iteration % args.log_interval
            if normalizer == 0:
                normalizer = args.log_interval
            timers.write(timers_to_log, writer, iteration,
                         normalizer=normalizer)

        if iteration % args.log_interval == 0:
            avg_lm_loss = total_lm_loss.item() / args.log_interval
            elapsed_time = timers('interval time').elapsed()
            if writer and args.rank == 0:
                writer.add_scalar('iteration_time',
                                  elapsed_time / args.log_interval, iteration)
            log_string = ' iteration {:8d}/{:8d} |'.format(iteration,
                                                            args.train_iters)
            log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
                elapsed_time * 1000.0 / args.log_interval)
            log_string += ' learning rate {:.3E} |'.format(learning_rate)
            log_string += ' lm loss {:.6E} |'.format(avg_lm_loss)
            if args.fp16:
                log_string += ' loss scale {:.1f} |'.format(
                    optimizer.loss_scale)
            print_rank_0(log_string)
            total_lm_loss = 0.0
            if report_memory_flag:
                report_memory('after {} iterations'.format(iteration))
                report_memory_flag = False
            timers.log(timers_to_log, normalizer=args.log_interval)

        # Autoresume
        if (iteration % args.adlr_autoresume_interval == 0) and args.adlr_autoresume:
            check_adlr_autoresume_termination(iteration, model, optimizer,
                                              lr_scheduler, args)

        # Checkpointing
        if args.save and args.save_interval and iteration % args.save_interval == 0:
            save_checkpoint(iteration, model, optimizer, lr_scheduler, args)

        # Evaluation
        if args.eval_interval and iteration % args.eval_interval == 0 and args.do_valid:
            prefix = 'iteration {}'.format(iteration)
            evaluate_and_print_results(prefix, val_data_iterator, model, args,
                                       writer, iteration, timers, False)

        if args.exit_interval and iteration % args.exit_interval == 0:
            torch.distributed.barrier()
            time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            rank = torch.distributed.get_rank()
            print('rank: {} | time: {} | exiting the program at iteration {}'.
                  format(rank, time_str, iteration), flush=True)
            exit()

    return iteration, skipped_iters


def evaluate(data_iterator, model, args, timers, verbose=False):
    """Evaluation."""

    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_lm_loss = 0

    with torch.no_grad():
        iteration = 0
        while iteration < args.eval_iters:
            iteration += 1
            if verbose and iteration % args.log_interval == 0:
                print_rank_0('Evaluating iter {}/{}'.format(iteration, args.eval_iters))
            # Forward evaluation.
            lm_loss = forward_step(data_iterator, model, args, timers)
            # Reduce across processes.
            if isinstance(model, args.DDP_type):
                torch.distributed.all_reduce(lm_loss.data)
                lm_loss.data = lm_loss.data / args.world_size

            total_lm_loss += lm_loss.data.detach().float().item()

    # Move model back to the train mode.
    model.train()

    total_lm_loss /= args.eval_iters
    return total_lm_loss


def evaluate_and_print_results(prefix, data_iterator, model,
                               args, writer, iteration,
                               timers, verbose=False):
    """Helper function to evaluate and dump results on screen."""
    lm_loss = evaluate(data_iterator, model, args, timers, verbose)
    lm_ppl = math.exp(min(20, lm_loss))
    print_rank_0('-' * 100)
    string = ' validation loss at {} | '.format(prefix)
    string += 'LM loss: {:.6E} | '.format(lm_loss)
    string += 'LM PPL: {:.6E}'.format(lm_ppl)
    length = len(string) + 1
    print_rank_0('-' * length)
    print_rank_0(string)
    print_rank_0('-' * length)

    if writer and args.rank == 0:
        writer.add_scalar('val_loss', lm_loss, iteration)
        writer.add_scalar('val_ppl', lm_ppl, iteration)

    return lm_loss


def get_train_val_test_data(args):
    """Load the data on rank zero and boradcast number of tokens to all GPUS."""

    (train_data, val_data, test_data) = (None, None, None)

    # Data loader only on rank 0 of each model parallel group.
    if mpu.get_model_parallel_rank() == 0:
        if args.use_npy_data_loader:
            (train_data, val_data, test_data), num_tokens, \
                eod_token = make_gpt2_dataloaders(args)
        else:
            data_config = configure_data()
            data_config.set_defaults(data_set_type='GPT2', transpose=False)
            (train_data, val_data, test_data), tokenizer = data_config.apply(
                args)
            num_tokens = tokenizer.num_tokens
            eod_token = tokenizer.get_command('eos').Id
            assert eod_token == tokenizer.get_command('pad').Id
        before = num_tokens
        after = before
        multiple = args.make_vocab_size_divisible_by * \
                   mpu.get_model_parallel_world_size()
        while (after % multiple) != 0:
            after += 1
        print_rank_0('> padded vocab (size: {}) with {} dummy '
                     'tokens (new size: {})'.format(
                         before, after - before, after))
        print_rank_0('> found end-of-document token: {}'.format(eod_token))
        token_counts = torch.cuda.LongTensor([after, eod_token, int(args.do_train), int(args.do_valid), int(args.do_test)])
    else:
        token_counts = torch.cuda.LongTensor([0, 0, 0, 0, 0])

    # Broadcast num tokens.
    torch.distributed.broadcast(token_counts,
                                mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    num_tokens = token_counts[0].item()
    eod_token = token_counts[1].item()
    args.do_train = token_counts[2].item()
    args.do_valid = token_counts[3].item()
    args.do_test = token_counts[4].item()

    return train_data, val_data, test_data, num_tokens, eod_token


def main():
    """Main training program."""

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Timer.
    timers = Timers()

    # Arguments.
    args = get_args()

    writer = None
    if args.tensorboard_dir and args.rank == 0:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir = args.tensorboard_dir)
        except ModuleNotFoundError:
            print_rank_0('WARNING: TensorBoard writing requested but is not '
                         'available (are you using PyTorch 1.1.0 or later?), '
                         'no TensorBoard logs will be written.')
            writer = None

    # Pytorch distributed.
    initialize_distributed(args)
    if torch.distributed.get_rank() == 0:
        print('Pretrain GPT2 model')
        print_args(args, writer)

    # Autoresume.
    torch.distributed.barrier()
    if args.adlr_autoresume:
        enable_adlr_autoresume(args)

    # Random seeds for reproducability.
    set_random_seed(args.seed)

    # Data stuff.
    train_data, val_data, test_data, args.vocab_size, \
        args.eod_token = get_train_val_test_data(args)

    # Model, optimizer, and learning rate.
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args)

    # Resume data loader if necessary.
    if args.resume_dataloader:
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

    #TODO: figure out how to properly set this especially when resuming training
    iteration = 0
    if args.train_iters > 0:
        if args.do_train:
            iteration, skipped = train(model, optimizer,
                                       lr_scheduler,
                                       train_data_iterator,
                                       val_data_iterator,
                                       timers, args, writer)

        if args.do_valid:
            prefix = 'the end of training for val data'
            val_loss = evaluate_and_print_results(prefix, val_data_iterator,
                                                  model, args, writer, iteration,
                                                  timers, False)

    if args.save and iteration != 0:
        save_checkpoint(iteration, model, optimizer,
                        lr_scheduler, args)

    if test_data is not None:
        test_data_iterator = iter(test_data)
    else:
        test_data_iterator = None

    if args.do_test:
        # Run on test data.
        prefix = 'the end of training for test data'
        evaluate_and_print_results(prefix, test_data_iterator,
                                   model, args, None, 0, timers, True)


if __name__ == "__main__":
    main()
