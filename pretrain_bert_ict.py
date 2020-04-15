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

"""Pretrain BERT for Inverse Cloze Task"""

import torch
import torch.nn.functional as F

from megatron import get_args
from megatron import get_timers
from megatron import mpu
from megatron import print_rank_0
from megatron.data.bert_dataset import build_train_valid_test_datasets
from megatron.model import ICTBertModel
from megatron.training import pretrain
from megatron.utils import make_data_loader
from megatron.utils import reduce_losses

num_batches = 0

def model_provider():
    """Build the model."""
    args = get_args()
    print_rank_0('building BERT models ...')

    model = ICTBertModel(
        ict_head_size=128,
        num_tokentypes=2,
        parallel_output=True)

    return model


def get_batch(data_iterator):

    # Items and their type.
    keys = ['input_text', 'input_types', 'input_pad_mask',
            'context_text', 'context_types', 'context_pad_mask']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is None:
        data = None
    else:
        data = next(data_iterator)
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    input_tokens = data_b['input_text'].long()
    input_types = data_b['input_types'].long()
    input_pad_mask = data_b['input_pad_mask'].long()
    context_tokens = data_b['context_text'].long()
    context_types = data_b['context_types'].long()
    context_pad_mask = data_b['context_pad_mask'].long()

    return input_tokens, input_types, input_pad_mask,\
           context_tokens, context_types, context_pad_mask


def forward_step(data_iterator, model):
    """Forward step."""
    timers = get_timers()

    # Get the batch.
    timers('batch generator').start()
    input_tokens, input_types, input_pad_mask,\
    context_tokens, context_types, context_pad_mask = get_batch(data_iterator)
    timers('batch generator').stop()

    # Forward model.
    retrieval_scores = model(input_tokens, input_pad_mask, input_types,
                             context_tokens, context_pad_mask, context_types).float()

    softmaxed = F.softmax(retrieval_scores, dim=1)
    top5_vals, top5_indices = torch.topk(softmaxed, k=5, sorted=True)
    batch_size = softmaxed.shape[0]

    top1_acc = torch.cuda.FloatTensor([sum([int(top5_indices[i, 0] == i) for i in range(batch_size)]) / batch_size])
    top5_acc = torch.cuda.FloatTensor([sum([int(i in top5_indices[i]) for i in range(batch_size)]) / batch_size])

    retrieval_loss = F.cross_entropy(softmaxed, torch.arange(batch_size).cuda())
    reduced_losses = reduce_losses([retrieval_loss, top1_acc, top5_acc])

    return retrieval_loss, {'retrieval loss': reduced_losses[0],
                            'top1_acc': reduced_losses[1],
                            'top5_acc': reduced_losses[2]}


def get_train_val_test_data():
    """Load the data on rank zero and boradcast number of tokens to all GPUS."""
    args = get_args()

    (train_data, valid_data, test_data) = (None, None, None)

    # Data loader only on rank 0 of each model parallel group.
    if mpu.get_model_parallel_rank() == 0:
        print_rank_0('> building train, validation, and test datasets '
                     'for BERT ...')

        data_parallel_size = mpu.get_data_parallel_world_size()
        data_parallel_rank = mpu.get_data_parallel_rank()
        global_batch_size = args.batch_size * data_parallel_size

        # Number of train/valid/test samples.
        train_iters = args.train_iters
        eval_iters = (train_iters // args.eval_iters + 1) * args.eval_iters
        test_iters = args.eval_iters
        train_val_test_num_samples = [train_iters * global_batch_size,
                                      eval_iters * global_batch_size,
                                      test_iters * global_batch_size]
        print_rank_0(' > datasets target sizes (minimum size):')
        print_rank_0('    train:      {}'.format(train_val_test_num_samples[0]))
        print_rank_0('    validation: {}'.format(train_val_test_num_samples[1]))
        print_rank_0('    test:       {}'.format(train_val_test_num_samples[2]))

        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
            data_prefix=args.data_path,
            data_impl=args.data_impl,
            splits_string=args.split,
            train_valid_test_num_samples=train_val_test_num_samples,
            max_seq_length=args.seq_length,
            masked_lm_prob=args.mask_prob,
            short_seq_prob=args.short_seq_prob,
            seed=args.seed,
            skip_warmup=(not args.mmap_warmup),
            ict_dataset=True)
        print_rank_0("> finished creating BERT ICT datasets ...")

        train_data = make_data_loader(train_ds)
        valid_data = make_data_loader(valid_ds)
        test_data = make_data_loader(test_ds)

        do_train = train_data is not None and args.train_iters > 0
        do_valid = valid_data is not None and args.eval_iters > 0
        do_test = test_data is not None and args.eval_iters > 0
        # Need to broadcast num_tokens and num_type_tokens.
        flags = torch.cuda.LongTensor(
            [int(do_train), int(do_valid), int(do_test)])
    else:
        flags = torch.cuda.LongTensor([0, 0, 0])

    # Broadcast num tokens.
    torch.distributed.broadcast(flags,
                                mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    args.do_train = flags[0].item()
    args.do_valid = flags[1].item()
    args.do_test = flags[2].item()

    return train_data, valid_data, test_data


if __name__ == "__main__":

    pretrain(get_train_val_test_data, model_provider, forward_step,
             args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'})
