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

"""Pretrain ALBERT"""

import torch
import torch.nn.functional as F

from megatron import mpu
from megatron.model import BertModel
from megatron.utils import print_rank_0
from megatron.utils import reduce_losses
from megatron.utils import vocab_size_with_padding
from megatron.training import run
from megatron.data.albert_dataset import build_train_valid_test_datasets
from megatron.data_utils.samplers import DistributedBatchSampler


def model_provider(args):
    """Build the model."""

    print_rank_0('building BERT model ...')

    model = BertModel(
        num_layers=args.num_layers,
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        embedding_dropout_prob=args.hidden_dropout,
        attention_dropout_prob=args.attention_dropout,
        output_dropout_prob=args.hidden_dropout,
        max_sequence_length=args.max_position_embeddings,
        checkpoint_activations=args.checkpoint_activations,
        checkpoint_num_layers=args.checkpoint_num_layers,
        add_binary_head=True,
        layernorm_epsilon=args.layernorm_epsilon,
        num_tokentypes=args.tokentype_size,
        parallel_output=True)

    return model


def get_batch(data_iterator, timers):

    # Items and their type.
    keys = ['text', 'types', 'labels', 'is_random', 'loss_mask', 'padding_mask']
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
    tokens = data_b['text'].long()
    types = data_b['types'].long()
    sentence_order = data_b['is_random'].long()
    loss_mask = data_b['loss_mask'].float()
    lm_labels = data_b['labels'].long()
    padding_mask = data_b['padding_mask'].long()

    return tokens, types, sentence_order, loss_mask, lm_labels, padding_mask


def forward_step(data_iterator, model, args, timers):
    """Forward step."""

    # Get the batch.
    timers('batch generator').start()
    tokens, types, sentence_order, loss_mask, lm_labels, padding_mask \
        = get_batch(data_iterator, timers)
    timers('batch generator').stop()

    # Forward model.
    lm_logits, sop_logits = model(tokens, padding_mask, tokentype_ids=types)

    sop_loss = F.cross_entropy(sop_logits.view(-1, 2).contiguous().float(),
                               sentence_order.view(-1).contiguous(),
                               ignore_index=-1)

    lm_loss_ = mpu.vocab_parallel_cross_entropy(lm_logits.contiguous().float(),
                                                lm_labels.contiguous())
    lm_loss = torch.sum(
        lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()

    loss = lm_loss + sop_loss

    reduced_losses = reduce_losses([lm_loss, sop_loss])

    return loss, {'lm loss': reduced_losses[0], 'sop loss': reduced_losses[1]}


def get_train_val_test_data(args):
    """Load the data on rank zero and boradcast number of tokens to all GPUS."""

    (train_data, valid_data, test_data) = (None, None, None)

    # Data loader only on rank 0 of each model parallel group.
    if mpu.get_model_parallel_rank() == 0:
        print_rank_0('> building train, validation, and test datasets '
                     'for ALBERT ...')

        if args.data_loader is None:
            args.data_loader = 'binary'
        if args.data_loader != 'binary':
            print('Unsupported {} data loader for ALBERT.'.format(
                args.data_loader))
            exit(1)
        if not args.data_path:
            print('ALBERT only supports a unified dataset specified '
                  'with --data-path')
            exit(1)

        data_parallel_size = mpu.get_data_parallel_world_size()
        data_parallel_rank = mpu.get_data_parallel_rank()
        global_batch_size = args.batch_size * data_parallel_size

        # Number of train/valid/test samples.
        train_iters = args.train_iters
        eval_iters = (train_iters // args.eval_interval + 1) * args.eval_iters
        test_iters = args.eval_iters
        train_val_test_num_samples = [args.train_iters * global_batch_size,
                                      eval_iters * global_batch_size,
                                      test_iters * global_batch_size]
        print_rank_0(' > datasets target sizes (minimum size):')
        print_rank_0('    train:      {}'.format(train_val_test_num_samples[0]))
        print_rank_0('    validation: {}'.format(train_val_test_num_samples[1]))
        print_rank_0('    test:       {}'.format(train_val_test_num_samples[2]))

        assert len(args.data_path) == 1
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
            vocab_file=args.vocab,
            data_prefix=args.data_path[0],
            data_impl=args.data_impl,
            splits_string=args.split,
            train_valid_test_num_samples=train_val_test_num_samples,
            max_seq_length=args.seq_length,
            masked_lm_prob=args.mask_prob,
            short_seq_prob=args.short_seq_prob,
            seed=args.seed,
            skip_warmup=args.skip_mmap_warmup)
        print_rank_0("> finished creating ALBERT datasets ...")

        def make_data_loader_(dataset):
            if not dataset:
                return None
            # Use a simple sampler with distributed batch sampler.
            sampler = torch.utils.data.SequentialSampler(dataset)
            batch_sampler = DistributedBatchSampler(
                sampler=sampler,
                batch_size=global_batch_size,
                drop_last=True,
                rank=data_parallel_rank,
                world_size=data_parallel_size)
            # Torch dataloader.
            return torch.utils.data.DataLoader(dataset,
                                               batch_sampler=batch_sampler,
                                               num_workers=args.num_workers,
                                               pin_memory=True)

        train_data = make_data_loader_(train_ds)
        valid_data = make_data_loader_(valid_ds)
        test_data = make_data_loader_(test_ds)

        do_train = train_data is not None and args.train_iters > 0
        do_valid = valid_data is not None and args.eval_iters > 0
        do_test = test_data is not None and args.eval_iters > 0
        # Need to broadcast num_tokens and num_type_tokens.
        num_tokens = vocab_size_with_padding(train_ds.num_tokens(), args)
        token_counts = torch.cuda.LongTensor([num_tokens,
                                              2, # hard coded num_type_tokens
                                              int(do_train),
                                              int(do_valid),
                                              int(do_test)])
    else:
        token_counts = torch.cuda.LongTensor([0, 0, 0, 0, 0])

    # Broadcast num tokens.
    torch.distributed.broadcast(token_counts,
                                mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    args.vocab_size = token_counts[0].item()
    args.tokentype_size = token_counts[1].item()
    args.do_train = token_counts[2].item()
    args.do_valid = token_counts[3].item()
    args.do_test = token_counts[4].item()

    return train_data, valid_data, test_data


if __name__ == "__main__":

    run('Pretrain BERT model', get_train_val_test_data,
        model_provider, forward_step)
