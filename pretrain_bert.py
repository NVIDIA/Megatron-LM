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

"""Pretrain BERT"""

import os
import random
import numpy as np
import torch

from arguments import get_args
from configure_data import configure_data
from fp16 import FP16_Module
from fp16 import FP16_Optimizer
from learning_rates import AnnealingLR
from model import BertModel
from model import get_params_for_weight_decay_optimization
from model import DistributedDataParallel as DDP
from optim import Adam
from utils import Timers
from utils import save_checkpoint
from utils import load_checkpoint


def get_model(tokenizer, args):
    """Build the model."""

    print('building BERT model ...')
    model = BertModel(tokenizer, args)
    print(' > number of parameters: {}'.format(
        sum([p.nelement() for p in model.parameters()])), flush=True)

    # GPU allocation.
    model.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16:
        model = FP16_Module(model)
        if args.fp32_embedding:
            model.module.model.bert.embeddings.word_embeddings.float()
            model.module.model.bert.embeddings.position_embeddings.float()
            model.module.model.bert.embeddings.token_type_embeddings.float()
        if args.fp32_tokentypes:
            model.module.model.bert.embeddings.token_type_embeddings.float()
        if args.fp32_layernorm:
            for name, _module in model.named_modules():
                if 'LayerNorm' in name:
                    _module.float()

    # Wrap model for distributed training.
    if args.world_size > 1:
        model = DDP(model)

    return model


def get_optimizer(model, args):
    """Set up the optimizer."""

    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, (DDP, FP16_Module)):
        model = model.module
    layers = model.model.bert.encoder.layer
    pooler = model.model.bert.pooler
    lmheads = model.model.cls.predictions
    nspheads = model.model.cls.seq_relationship
    embeddings = model.model.bert.embeddings
    param_groups = []
    param_groups += list(get_params_for_weight_decay_optimization(layers))
    param_groups += list(get_params_for_weight_decay_optimization(pooler))
    param_groups += list(get_params_for_weight_decay_optimization(nspheads))
    param_groups += list(get_params_for_weight_decay_optimization(embeddings))
    param_groups += list(get_params_for_weight_decay_optimization(
        lmheads.transform))
    param_groups[1]['params'].append(lmheads.bias)

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
        num_iters = args.train_iters * args.epochs
    init_step = -1
    warmup_iter = args.warmup * num_iters
    lr_scheduler = AnnealingLR(optimizer,
                               start_lr=args.lr,
                               warmup_iter=warmup_iter,
                               num_iters=num_iters,
                               decay_style=args.lr_decay_style,
                               last_iter=init_step)

    return lr_scheduler


def setup_model_and_optimizer(args, tokenizer):
    """Setup model and optimizer."""

    model = get_model(tokenizer, args)
    optimizer = get_optimizer(model, args)
    lr_scheduler = get_learning_rate_scheduler(optimizer, args)
    criterion = torch.nn.CrossEntropyLoss(reduce=False, ignore_index=-1)

    if args.load is not None:
        epoch, i, total_iters = load_checkpoint(model, optimizer,
                                                lr_scheduler, args)
        if args.resume_dataloader:
            args.epoch = epoch
            args.mid_epoch_iters = i
            args.total_iters = total_iters

    return model, optimizer, lr_scheduler, criterion


def get_batch(data):
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
    tokens = torch.autograd.Variable(data['text'].long())
    types = torch.autograd.Variable(data['types'].long())
    next_sentence = torch.autograd.Variable(data['is_random'].long())
    loss_mask = torch.autograd.Variable(data['mask'].float())
    lm_labels = torch.autograd.Variable(data['mask_labels'].long())
    padding_mask = torch.autograd.Variable(data['pad_mask'].byte())
    # Move to cuda
    tokens = tokens.cuda()
    types = types.cuda()
    next_sentence = next_sentence.cuda()
    loss_mask = loss_mask.cuda()
    lm_labels = lm_labels.cuda()
    padding_mask = padding_mask.cuda()

    return tokens, types, next_sentence, loss_mask, lm_labels, padding_mask


def forward_step(data, model, criterion, args):
    """Forward step."""

    # Get the batch.
    tokens, types, next_sentence, loss_mask, lm_labels, \
        padding_mask = get_batch(data)
    # Forward model.
    output, nsp = model(tokens, types, 1-padding_mask,
                        checkpoint_activations=args.checkpoint_activations)
    nsp_loss = criterion(nsp.view(-1, 2).contiguous().float(),
                         next_sentence.view(-1).contiguous()).mean()
    losses = criterion(output.view(-1, args.data_size).contiguous().float(),
                       lm_labels.contiguous().view(-1).contiguous())
    loss_mask = loss_mask.contiguous()
    loss_mask = loss_mask.view(-1)
    lm_loss = torch.sum(
        losses * loss_mask.view(-1).float()) / loss_mask.sum()

    return lm_loss, nsp_loss


def backward_step(optimizer, model, lm_loss, nsp_loss, args):
    """Backward step."""

    # Total loss.
    loss = lm_loss + nsp_loss

    # Backward pass.
    optimizer.zero_grad()
    if args.fp16:
        optimizer.backward(loss, update_master_grads=False)
    else:
        loss.backward()

    # Reduce across processes.
    lm_loss_reduced = lm_loss
    nsp_loss_reduced = nsp_loss
    if args.world_size > 1:
        reduced_losses = torch.cat((lm_loss.view(1), nsp_loss.view(1)))
        torch.distributed.all_reduce(reduced_losses.data)
        reduced_losses.data = reduced_losses.data / args.world_size
        model.allreduce_params(reduce_after=False,
                               fp32_allreduce=args.fp32_allreduce)
        lm_loss_reduced = reduced_losses[0]
        nsp_loss_reduced = reduced_losses[1]

    # Update master gradients.
    if args.fp16:
        optimizer.update_master_grads()

    # Clipping gradients helps prevent the exploding gradient.
    if args.clip_grad > 0:
        if not args.fp16:
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_grad)
        else:
            optimizer.clip_master_grads(args.clip_grad)

    return lm_loss_reduced, nsp_loss_reduced


def train_step(input_data, model, criterion, optimizer, lr_scheduler, args):
    """Single training step."""

    # Forward model for one step.
    lm_loss, nsp_loss = forward_step(input_data, model, criterion, args)

    # Calculate gradients, reduce across processes, and clip.
    lm_loss_reduced, nsp_loss_reduced = backward_step(optimizer, model, lm_loss,
                                                      nsp_loss, args)

    # Update parameters.
    optimizer.step()

    # Update learning rate.
    skipped_iter = 0
    if not (args.fp16 and optimizer.overflow):
        lr_scheduler.step()
    else:
        skipped_iter = 1

    return lm_loss_reduced, nsp_loss_reduced, skipped_iter


def train_epoch(epoch, model, optimizer, train_data,
                lr_scheduler, criterion, timers, args):
    """Train one full epoch."""

    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    total_lm_loss = 0.0
    total_nsp_loss = 0.0

    # Iterations.
    max_iters = args.train_iters
    iteration = 0
    skipped_iters = 0
    if args.resume_dataloader:
        iteration = args.mid_epoch_iters
        args.resume_dataloader = False

    # Data iterator.
    data_iterator = iter(train_data)

    timers('interval time').start()
    while iteration < max_iters:

        lm_loss, nsp_loss, skipped_iter = train_step(next(data_iterator),
                                                     model,
                                                     criterion,
                                                     optimizer,
                                                     lr_scheduler,
                                                     args)
        skipped_iters += skipped_iter
        iteration += 1

        # Update losses.
        total_lm_loss += lm_loss.data.detach().float()
        total_nsp_loss += nsp_loss.data.detach().float()

        # Logging.
        if iteration % args.log_interval == 0:
            learning_rate = optimizer.param_groups[0]['lr']
            avg_nsp_loss = total_nsp_loss.item() / args.log_interval
            avg_lm_loss = total_lm_loss.item() / args.log_interval
            elapsed_time = timers('interval time').elapsed()
            log_string = ' epoch{:2d} |'.format(epoch)
            log_string += ' iteration {:8d}/{:8d} |'.format(iteration,
                                                            max_iters)
            log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
                elapsed_time * 1000.0 / args.log_interval)
            log_string += ' learning rate {:.3E} |'.format(learning_rate)
            log_string += ' lm loss {:.3E} |'.format(avg_lm_loss)
            log_string += ' nsp loss {:.3E} |'.format(avg_nsp_loss)
            if args.fp16:
                log_string += ' loss scale {:.1f} |'.format(
                    optimizer.loss_scale)
            print(log_string, flush=True)
            total_nsp_loss = 0.0
            total_lm_loss = 0.0

        # Checkpointing
        if args.save and args.save_iters and iteration % args.save_iters == 0:
            total_iters = args.train_iters * (epoch-1) + iteration
            model_suffix = 'model/%d.pt' % (total_iters)
            save_checkpoint(model_suffix, epoch, iteration, model, optimizer,
                            lr_scheduler, args)

    return iteration, skipped_iters


def evaluate(data_source, model, criterion, args):
    """Evaluation."""

    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_lm_loss = 0
    total_nsp_loss = 0
    max_iters = args.eval_iters

    with torch.no_grad():
        data_iterator = iter(data_source)
        iteration = 0
        while iteration < max_iters:
            # Forward evaluation.
            lm_loss, nsp_loss = forward_step(next(data_iterator), model,
                                             criterion, args)
            # Reduce across processes.
            if isinstance(model, DDP):
                reduced_losses = torch.cat((lm_loss.view(1), nsp_loss.view(1)))
                torch.distributed.all_reduce(reduced_losses.data)
                reduced_losses.data = reduced_losses.data/args.world_size
                lm_loss = reduced_losses[0]
                nsp_loss = reduced_losses[1]

            total_lm_loss += lm_loss.data.detach().float().item()
            total_nsp_loss += nsp_loss.data.detach().float().item()
            iteration += 1

    # Move model back to the train mode.
    model.train()

    total_lm_loss /= max_iters
    total_nsp_loss /= max_iters
    return total_lm_loss, total_nsp_loss


def initialize_distributed(args):
    """Initialize torch.distributed."""

    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()
    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device)
    # Call the init process
    if args.world_size > 1:
        init_method = 'tcp://'
        master_ip = os.getenv('MASTER_ADDR', 'localhost')
        master_port = os.getenv('MASTER_PORT', '6000')
        init_method += master_ip + ':' + master_port
        torch.distributed.init_process_group(
            backend=args.distributed_backend,
            world_size=args.world_size, rank=args.rank,
            init_method=init_method)


def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


def main():
    """Main training program."""

    print('Pretrain BERT model')

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Timer.
    timers = Timers()

    # Arguments.
    args = get_args()

    # Pytorch distributed.
    initialize_distributed(args)

    # Random seeds for reproducability.
    set_random_seed(args.seed)

    # Data stuff.
    data_config = configure_data()
    data_config.set_defaults(data_set_type='BERT', transpose=False)
    (train_data, val_data, test_data), tokenizer = data_config.apply(args)
    args.data_size = tokenizer.num_tokens

    # Model, optimizer, and learning rate.
    model, optimizer, lr_scheduler, criterion = setup_model_and_optimizer(
        args, tokenizer)

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        total_iters = 0
        skipped_iters = 0
        start_epoch = 1
        best_val_loss = float('inf')
        # Resume data loader if necessary.
        if args.resume_dataloader:
            start_epoch = args.epoch
            total_iters = args.total_iters
            train_data.batch_sampler.start_iter = total_iters % len(train_data)
        # For all epochs.
        for epoch in range(start_epoch, args.epochs+1):
            if args.shuffle:
                train_data.batch_sampler.sampler.set_epoch(epoch+args.seed)
            timers('epoch time').start()
            iteration, skipped = train_epoch(epoch, model, optimizer,
                                             train_data, lr_scheduler,
                                             criterion, timers, args)
            elapsed_time = timers('epoch time').elapsed()
            total_iters += iteration
            skipped_iters += skipped
            lm_loss, nsp_loss = evaluate(val_data, model, criterion, args)
            val_loss = lm_loss + nsp_loss
            print('-' * 100)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:.4E} | '
                  'valid LM Loss {:.4E} | valid NSP Loss {:.4E}'.format(
                      epoch, elapsed_time, val_loss, lm_loss, nsp_loss))
            print('-' * 100)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if args.save:
                    best_path = 'best/model.pt'
                    print('saving best model to:',
                           os.path.join(args.save, best_path))
                    save_checkpoint(best_path, epoch+1, total_iters, model,
                                    optimizer, lr_scheduler, args)


    except KeyboardInterrupt:
        print('-' * 100)
        print('Exiting from training early')
        if args.save:
            cur_path = 'current/model.pt'
            print('saving current model to:',
                   os.path.join(args.save, cur_path))
            save_checkpoint(cur_path, epoch, total_iters, model, optimizer,
                            lr_scheduler, args)
        exit()

    if args.save:
        final_path = 'final/model.pt'
        print('saving final model to:', os.path.join(args.save, final_path))
        save_checkpoint(final_path, args.epochs, total_iters, model, optimizer,
                        lr_scheduler, args)

    if test_data is not None:
        # Run on test data.
        print('entering test')
        lm_loss, nsp_loss = evaluate(test_data, model, criterion, args)
        test_loss = lm_loss + nsp_loss
        print('=' * 100)
        print('| End of training | test loss {:5.4f} | valid LM Loss {:.4E} |'
              ' valid NSP Loss {:.4E}'.format(test_loss, lm_loss, nsp_loss))
        print('=' * 100)


if __name__ == "__main__":
    main()
