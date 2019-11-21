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

import torch
import torch.nn.functional as F

from configure_data import configure_data
from megatron import mpu
from megatron.model import BertModel
from megatron.utils import print_rank_0
from megatron.utils import reduce_losses
from megatron.utils import vocab_size_with_padding
from megatron.training import run
from megatron.data import AlbertDataset, split_dataset
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

    (train_data, val_data, test_data) = (None, None, None)

    # Data loader only on rank 0 of each model parallel group.
    if mpu.get_model_parallel_rank() == 0:
        if args.data_loader == None:
            args.data_loader = 'binary'
        if args.data_loader == 'binary':
            if not args.max_num_samples:
                args.max_num_samples = (args.train_iters + 2 * args.eval_iters) * args.batch_size
            if not args.data_path:
                print("Albert currently only supports a unified dataset specified with --data-path")
                exit(1)
            print_rank_0("Creating AlbertDataset...")
            full_data = AlbertDataset(
                vocab_file=args.vocab,
                data_prefix=args.data_path,
                data_impl=args.data_impl,
                skip_warmup=args.skip_mmap_warmup,
                num_epochs=args.data_epochs,
                max_num_samples=args.max_num_samples,
                masked_lm_prob=args.mask_prob,
                max_seq_length=args.seq_length,
                short_seq_prob=args.short_seq_prob,
                seed=args.seed)
            print_rank_0("Finished creating AlbertDataset...")
            split = split_dataset.get_split(args)
            if split_dataset.should_split(split):
                train_ds, val_ds, test_ds = split_dataset.split_ds(full_data, split, args.shuffle)
            else:
                train_ds = full_data
            num_tokens = train_ds.num_tokens()

            world_size = mpu.get_data_parallel_world_size()
            rank = mpu.get_data_parallel_rank()
            global_batch_size = args.batch_size * world_size
            num_workers = args.num_workers

            def make_data_loader_(dataset):
                if not dataset:
                    return None
                # Use a simple sampler with distributed batch sampler.
                sampler = torch.utils.data.SequentialSampler(dataset)
                batch_sampler = DistributedBatchSampler(
                    sampler=sampler,
                    batch_size=global_batch_size,
                    drop_last=True,
                    rank=rank,
                    world_size=world_size)
                # Torch dataloader.
                return torch.utils.data.DataLoader(dataset,
                                                   batch_sampler=batch_sampler,
                                                   num_workers=num_workers,
                                                   pin_memory=True)

            train_data = make_data_loader_(train_ds)
            valid_data = make_data_loader_(val_ds)
            test_data = make_data_loader_(test_ds)

            do_train = train_data is not None and args.train_iters > 0
            do_valid = valid_data is not None and args.eval_iters > 0
            do_test = test_data is not None and args.eval_iters > 0
            # Need to broadcast num_tokens and num_type_tokens.
            token_counts = torch.cuda.LongTensor([num_tokens,
                                                  2, # hard coded num_type_tokens for now
                                                  int(do_train),
                                                  int(do_valid),
                                                  int(do_test)])
        else:
            print("Unsupported data loader for BERT.")
            exit(1)
    else:
        token_counts = torch.cuda.LongTensor([0, 0, 0, 0, 0])


    # Broadcast num tokens.
    torch.distributed.broadcast(token_counts,
                                mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    num_tokens = token_counts[0].item()
    num_type_tokens = token_counts[1].item()
    args.do_train = token_counts[2].item()
    args.do_valid = token_counts[3].item()
    args.do_test = token_counts[4].item()

    args.vocab_size = num_tokens
    args.tokentype_size = num_type_tokens

    return train_data, val_data, test_data


if __name__ == "__main__":

    run('Pretrain BERT model', get_train_val_test_data,
        model_provider, forward_step)
