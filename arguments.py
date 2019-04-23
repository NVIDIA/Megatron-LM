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

"""argparser configuration"""

import argparse
import os
import torch


def add_model_config_args(parser):
    """Model arguments"""

    group = parser.add_argument_group('model', 'model configuration')

    group.add_argument('--pretrained-bert', action='store_true',
                       help='use a pretrained bert-large-uncased model instead'
                       'of initializing from scratch. See '
                       '--tokenizer-model-type to specify which pretrained '
                       'BERT model to use')
    group.add_argument('--attention-dropout', type=float, default=0.1,
                       help='dropout probability for attention weights')
    group.add_argument('--num-attention-heads', type=int, default=16,
                       help='num of transformer attention heads')
    group.add_argument('--hidden-size', type=int, default=1024,
                       help='tansformer hidden size')
    group.add_argument('--intermediate-size', type=int, default=None,
                       help='transformer embedding dimension for FFN'
                       'set to 4*`--hidden-size` if it is None')
    group.add_argument('--num-layers', type=int, default=24,
                       help='num decoder layers')
    group.add_argument('--layernorm-epsilon', type=float, default=1e-12,
                       help='layer norm epsilon')
    group.add_argument('--hidden-dropout', type=float, default=0.0,
                       help='dropout probability for hidden state transformer')
    group.add_argument('--max-position-embeddings', type=int, default=512,
                       help='maximum number of position embeddings to use')
    group.add_argument('--vocab-size', type=int, default=30522,
                       help='vocab size to use for non-character-level '
                       'tokenization. This value will only be used when '
                       'creating a tokenizer')

    return parser


def add_fp16_config_args(parser):
    """Mixed precision arguments."""

    group = parser.add_argument_group('fp16', 'fp16 configurations')

    group.add_argument('--fp16', action='store_true',
                       help='Run model in fp16 mode')
    group.add_argument('--fp32-embedding', action='store_true',
                       help='embedding in fp32')
    group.add_argument('--fp32-layernorm', action='store_true',
                       help='layer norm in fp32')
    group.add_argument('--fp32-tokentypes', action='store_true',
                       help='embedding token types in fp32')
    group.add_argument('--fp32-allreduce', action='store_true',
                       help='all-reduce in fp32')
    group.add_argument('--hysteresis', type=int, default=2,
                       help='hysteresis for dynamic loss scaling')
    group.add_argument('--loss-scale', type=float, default=None,
                       help='Static loss scaling, positive power of 2 '
                       'values can improve fp16 convergence. If None, dynamic'
                       'loss scaling is used.')
    group.add_argument('--loss-scale-window', type=float, default=1000,
                       help='Window over which to raise/lower dynamic scale')
    group.add_argument('--min-scale', type=float, default=1,
                       help='Minimum loss scale for dynamic loss scale')

    return parser


def add_training_args(parser):
    """Training arguments."""

    group = parser.add_argument_group('train', 'training configurations')

    group.add_argument('--batch-size', type=int, default=4,
                       help='Data Loader batch size')
    group.add_argument('--weight-decay', type=float, default=0.01,
                       help='weight decay coefficient for L2 regularization')
    group.add_argument('--checkpoint-activations', action='store_true',
                       help='checkpoint activation to allow for training '
                       'with larger models and sequences')
    group.add_argument('--clip-grad', type=float, default=1.0,
                       help='gradient clipping')
    group.add_argument('--epochs', type=int, default=1,
                       help='upper epoch limit')
    group.add_argument('--log-interval', type=int, default=100,
                       help='report interval')
    group.add_argument('--train-iters', type=int, default=1000000,
                       help='number of iterations per epoch')
    group.add_argument('--seed', type=int, default=1234,
                       help='random seed')
    # Learning rate.
    group.add_argument('--lr-decay-iters', type=int, default=None,
                       help='number of iterations to decay LR over,'
                       ' If None defaults to `--train-iters`*`--epochs`')
    group.add_argument('--lr-decay-style', type=str, default='linear',
                       choices=['constant', 'linear', 'cosine', 'exponential'],
                       help='learning rate decay function')
    group.add_argument('--lr', type=float, default=1.0e-4,
                       help='initial learning rate')
    group.add_argument('--warmup', type=float, default=0.01,
                       help='percentage of data to warmup on (.01 = 1% of all '
                       'training iters). Default 0.01')
    # model checkpointing
    group.add_argument('--save', type=str, default=None,
                       help='Output directory to save checkpoints to.')
    group.add_argument('--save-iters', type=int, default=None,
                       help='Save every so often iterations.')
    group.add_argument('--save-optim', action='store_true',
                       help='Save current optimizer.')
    group.add_argument('--save-rng', action='store_true',
                       help='Save current rng state.')
    group.add_argument('--save-all-rng', action='store_true',
                       help='Save current rng state of each rank in '
                       'distributed training.')
    group.add_argument('--load', type=str, default=None,
                       help='Path to a particular model checkpoint. \
                             (ex. `savedir/model.1000.pt`)')
    group.add_argument('--load-optim', action='store_true',
                       help='Load most recent optimizer corresponding '
                       'to `--load`.')
    group.add_argument('--load-rng', action='store_true',
                       help='Load most recent rng state corresponding '
                       'to `--load`.')
    group.add_argument('--load-all-rng', action='store_true',
                       help='Load most recent rng state of each rank in '
                       'distributed training corresponding to `--load`('
                       'complementary to `--save-all-rng`).')
    group.add_argument('--resume-dataloader', action='store_true',
                       help='Resume the dataloader when resuming training. '
                       'Does not apply to tfrecords dataloader, try resuming'
                       'with a different seed in this case.')
    # distributed training args
    group.add_argument('--distributed-backend', default='nccl',
                       help='which backend to use for distributed '
                       'training. One of [gloo, nccl]')
    group.add_argument('--local_rank', type=int, default=None,
                       help='local rank passed from distributed launcher')

    return parser


def add_evaluation_args(parser):
    """Evaluation arguments."""

    group = parser.add_argument_group('validation', 'validation configurations')

    group.add_argument('--eval-batch-size', type=int, default=None,
                       help='Data Loader batch size for evaluation datasets.'
                       'Defaults to `--batch-size`')
    group.add_argument('--eval-iters', type=int, default=2000,
                       help='number of iterations per epoch to run '
                       'validation/test for')
    group.add_argument('--eval-seq-length', type=int, default=None,
                       help='Maximum sequence length to process for '
                       'evaluation. Defaults to `--seq-length`')
    group.add_argument('--eval-max-preds-per-seq', type=int, default=None,
                       help='Maximum number of predictions to use for '
                       'evaluation. Defaults to '
                       'math.ceil(`--eval-seq-length`*.15/10)*10')

    return parser


def add_data_args(parser):
    """Train/valid/test data arguments."""

    group = parser.add_argument_group('data', 'data configurations')

    group.add_argument('--shuffle', action='store_true',
                       help='Shuffle data. Shuffling is deterministic '
                       'based on seed and current epoch.')
    group.add_argument('--train-data', nargs='+', required=True,
                       help='Filename (or whitespace separated filenames) '
                       'for training.')
    group.add_argument('--delim', default=',',
                       help='delimiter used to parse csv data files')
    group.add_argument('--text-key', default='sentence',
                       help='key to use to extract text from json/csv')
    group.add_argument('--eval-text-key', default=None,
                       help='key to use to extract text from '
                       'json/csv evaluation datasets')
    group.add_argument('--valid-data', nargs='*', default=None,
                       help="""Filename for validation data.""")
    group.add_argument('--split', default='1000,1,1',
                       help='comma-separated list of proportions for training,'
                       ' validation, and test split')
    group.add_argument('--test-data', nargs='*', default=None,
                       help="""Filename for testing""")

    group.add_argument('--lazy-loader', action='store_true',
                       help='whether to lazy read the data set')
    group.add_argument('--loose-json', action='store_true',
                       help='Use loose json (one json-formatted string per '
                       'newline), instead of tight json (data file is one '
                       'json string)')
    group.add_argument('--presplit-sentences', action='store_true',
                       help='Dataset content consists of documents where '
                       'each document consists of newline separated sentences')
    group.add_argument('--num-workers', type=int, default=2,
                       help="""Number of workers to use for dataloading""")
    group.add_argument('--tokenizer-model-type', type=str,
                       default='bert-large-uncased',
                       help="Model type to use for sentencepiece tokenization \
                       (one of ['bpe', 'char', 'unigram', 'word']) or \
                       bert vocab to use for BertWordPieceTokenizer (one of \
                       ['bert-large-uncased', 'bert-large-cased', etc.])")
    group.add_argument('--tokenizer-path', type=str, default='tokenizer.model',
                       help='path used to save/load sentencepiece tokenization '
                       'models')
    group.add_argument('--tokenizer-type', type=str,
                       default='BertWordPieceTokenizer',
                       choices=['CharacterLevelTokenizer',
                                'SentencePieceTokenizer',
                                'BertWordPieceTokenizer'],
                       help='what type of tokenizer to use')
    group.add_argument("--cache-dir", default=None, type=str,
                       help="Where to store pre-trained BERT downloads")
    group.add_argument('--use-tfrecords', action='store_true',
                       help='load `--train-data`, `--valid-data`, '
                       '`--test-data` from BERT tf records instead of '
                       'normal data pipeline')
    group.add_argument('--seq-length', type=int, default=512,
                       help="Maximum sequence length to process")
    group.add_argument('--max-preds-per-seq', type=int, default=None,
                       help='Maximum number of predictions to use per sequence.'
                       'Defaults to math.ceil(`--seq-length`*.15/10)*10.'
                       'MUST BE SPECIFIED IF `--use-tfrecords` is True.')

    return parser


def print_args(args):
    """Print arguments."""

    print('arguments:', flush=True)
    for arg in vars(args):
        dots = '.' * (29 - len(arg))
        print('  {} {} {}'.format(arg, dots, getattr(args, arg)), flush=True)


def get_args():
    """Parse all the args."""

    parser = argparse.ArgumentParser(description='PyTorch BERT Model')
    parser = add_model_config_args(parser)
    parser = add_fp16_config_args(parser)
    parser = add_training_args(parser)
    parser = add_evaluation_args(parser)
    parser = add_data_args(parser)

    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()
    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))

    args.dynamic_loss_scale = False
    if args.loss_scale is None:
        args.dynamic_loss_scale = True
        print(' > using dynamic loss scaling')

    # The args fp32_* or fp16_* meant to be active when the
    # args fp16 is set. So the default behaviour should all
    # be false.
    if not args.fp16:
        args.fp32_embedding = False
        args.fp32_tokentypes = False
        args.fp32_layernorm = False

    print_args(args)
    return args
