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


_GLOBAL_ARGS = None



def parse_args(extra_args_provider=None):

    global _GLOBAL_ARGS
    assert _GLOBAL_ARGS is None, 'args already initializeed'
    _GLOBAL_ARGS = get_args_(extra_args_provider=extra_args_provider)
    return _GLOBAL_ARGS


def get_args(extra_args_provider=None):

    global _GLOBAL_ARGS
    if _GLOBAL_ARGS is None:
        return parse_args(extra_args_provider=extra_args_provider)
    else:
        return _GLOBAL_ARGS


def add_network_size_args(parser):
    group = parser.add_argument_group(title='network size')
    
    group.add_argument('--num-layers', type=int, required=True,
                       help='Number of transformer layers.')
    group.add_argument('--hidden-size', type=int, required=True,
                       help='Tansformer hidden size.')
    group.add_argument('--num-attention-heads', type=int, required=True,
                       help='Number of transformer attention heads.')
    group.add_argument('--max-position-embeddings', type=int, required=True,
                       help='Maximum number of position embeddings to use. '
                       'This is the size of position embedding.')
    group.add_argument('--make-vocab-size-divisible-by', type=int, default=128,
                       help='Pad the vocab size to be divisible by this value.'
                       'This is added for computational efficieny reasons.')
    
    return parser


def add_regularization_args(parser):
    group = parser.add_argument_group(title='regularization')

    group.add_argument('--attention-dropout', type=float, default=0.1,
                       help='Post attention dropout ptobability.')
    group.add_argument('--hidden-dropout', type=float, default=0.1,
                       help='Dropout probability for hidden state transformer.')
    group.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay coefficient for L2 regularization.')
    group.add_argument('--clip-grad', type=float, default=1.0,
                       help='Gradient clipping based on global L2 norm.')

    return parser
    

def add_training_args(parser):
    group = parser.add_argument_group(title='training')

    group.add_argument('--batch-size', type=int, required=True,
                       help='Batch size per model instance (local batch size). '
                       'Global batch size is local batch size times data '
                       'parallel size.')
    group.add_argument('--checkpoint-activations', action='store_true',
                       help='Checkpoint activation to allow for training '
                       'with larger models, sequences, and batch sizes.')
    group.add_argument('--checkpoint-num-layers', type=int, default=1,
                       help='chunk size (number of layers) for checkpointing.')
    group.add_argument('--train-iters', type=int, required=True,
                       help='Total number of iterations to train over all '
                       'training runs.')
    group.add_argument('--log-interval', type=int, default=100,
                       help='Report loss and timing interval.')
    group.add_argument('--exit-interval', type=int, default=None,
                       help='Exit the program after the iteration is divisible '
                       'by this value.')
    group.add_argument('--tensorboard-dir', type=str, default=None,
                       help='Write TensorBoard logs to this directory.')

    return parser


def add_initialization_args(parser):
    group = parser.add_argument_group(title='initialization')

    group.add_argument('--seed', type=int, default=1234,
                       help='Random seed used for python, numpy, '
                       'pytorch, and cuda.')
    group.add_argument('--init-method-std', type=float, default=0.02,
                       help='Standard deviation of the zero mean normal '
                       'distribution used for weight initialization.')
    
    return parser


def add_learning_rate_args(parser):
    group = parser.add_argument_group(title='learning rate')

    group.add_argument('--lr', type=float, required=True,
                       help='Initial learning rate. Depending on decay style '
                       'and initial warmup, the learing rate at each '
                       'iteration would be different.')
    group.add_argument('--lr-decay-style', type=str, default='linear',
                       choices=['constant', 'linear', 'cosine', 'exponential'],
                       help='Learning rate decay function.')
    group.add_argument('--lr-decay-iters', type=int, default=None,
                       help='number of iterations to decay learning rate over,'
                       ' If None defaults to `--train-iters`')
    group.add_argument('--min-lr', type=float, default=0.0,
                       help='Minumum value for learning rate. The scheduler'
                       'clip values below this threshold.')
    group.add_argument('--warmup', type=float, default=0.01,
                       help='Percentage of total iterations to warmup on '
                       '(.01 = 1 percent of all training iters).')
    group.add_argument('--override-lr-scheduler', action='store_true',
                       help='Reset the values of the scheduler (learning rate,'
                       'warmup iterations, minimum learning rate, maximum '
                       'number of iterations, and decay style from input '
                       'arguments and ignore values from checkpoints. Note'
                       'that all the above values will be reset.')
    group.add_argument('--use-checkpoint-lr-scheduler', action='store_true',
                       help='Use checkpoint to set the values of the scheduler '
                       '(learning rate, warmup iterations, minimum learning '
                       'rate, maximum number of iterations, and decay style '
                       'from checkpoint and ignore input arguments.')

    return parser


def add_checkpointing_args(parser):
    group = parser.add_argument_group(title='checkpointing')

    group.add_argument('--save', type=str, default=None,
                       help='Output directory to save checkpoints to.')
    group.add_argument('--save-interval', type=int, default=None,
                       help='Number of iterations between checkpoint saves.')
    group.add_argument('--no-save-optim', action='store_true',
                       help='Do not save current optimizer.')
    group.add_argument('--no-save-rng', action='store_true',
                       help='Do not save current rng state.')
    group.add_argument('--load', type=str, default=None,
                       help='Directory containing a model checkpoint.')
    group.add_argument('--no-load-optim', action='store_true',
                       help='Do not load optimizer when loading checkpoint.')
    group.add_argument('--no-load-rng', action='store_true',
                       help='Do not load rng state when loading checkpoint.')
    group.add_argument('--finetune', action='store_true',
                       help='Load model for finetuning. Do not load optimizer '
                       'or rng state from checkpoint and set iteration to 0. '
                       'Assumed when loading a release checkpoint.')

    return parser


def add_mixed_precision_args(parser):
    group = parser.add_argument_group(title='mixed precision')

    group.add_argument('--fp16', action='store_true',
                       help='Run model in fp16 mode.')
    group.add_argument('--apply-query-key-layer-scaling', action='store_true',
                       help='Scale Q * K^T by 1 / layer-number. If this flag '
                       'is set, then it will automatically set '
                       'attention-softmax-in-fp32 to true')
    group.add_argument('--attention-softmax-in-fp32', action='store_true',
                       help='Run attention masking and softmax in fp32.')
    group.add_argument('--hysteresis', type=int, default=2,
                       help='hysteresis for dynamic loss scaling')
    group.add_argument('--loss-scale', type=float, default=None,
                       help='Static loss scaling, positive power of 2 '
                       'values can improve fp16 convergence. If None, dynamic'
                       'loss scaling is used.')
    group.add_argument('--loss-scale-window', type=float, default=1000,
                       help='Window over which to raise/lower dynamic scale.')
    group.add_argument('--min-scale', type=float, default=1,
                       help='Minimum loss scale for dynamic loss scale.')

    return parser


def add_distributed_args(parser):
    group = parser.add_argument_group(title='mixed precision')

    group.add_argument('--model-parallel-size', type=int, default=1,
                       help='Size of the model parallel.')
    group.add_argument('--distributed-backend', default='nccl',
                       choices=['nccl', 'gloo'],
                       help='Which backend to use for distributed training.')
    group.add_argument('--DDP-impl', default='local',
                       choices=['local', 'torch'], 
                       help='which DistributedDataParallel implementation '
                       'to use.')
    group.add_argument('--local_rank', type=int, default=None,
                       help='local rank passed from distributed launcher.')

    return parser


def add_validation_args(parser):
    group = parser.add_argument_group(title='validation')

    group.add_argument('--eval-iters', type=int, default=100,
                       help='Number of iterations to run for evaluation'
                       'validation/test for.')
    group.add_argument('--eval-interval', type=int, default=1000,
                       help='Interval between running evaluation on '
                       'validation set.')

    return parser


def add_data_args(parser):
    group = parser.add_argument_group(title='data and dataloader')

    group.add_argument('--data-path', type=str, required=True,
                       help='Path to combined dataset to split.')
    group.add_argument('--split', type=str, required=True,
                       help='Comma-separated list of proportions for training,'
                       ' validation, and test split. For example the split '
                       '`90,5,5` will use 90% of data for training, 5% for '
                       'validation and 5% for test.')
    group.add_argument('--vocab-file', type=str, required=True,
                       help='Path to the vocab file.')
    group.add_argument('--seq-length', type=int, required=True,
                       help="Maximum sequence length to process.")
    group.add_argument('--mask-prob', type=float, default=0.15,
                       help='Probability of replacing a token with mask.')
    group.add_argument('--short-seq-prob', type=float, default=0.1,
                       help='Probability of producing a short sequence.')
    group.add_argument('--mmap-warmup', action='store_true',
                       help='Warm up mmap files.')
    group.add_argument('--num-workers', type=int, default=2,
                       help="Dataloader number of workers.")




    return parser

########################


def add_model_config_args(parser):
    """Model arguments"""
    
    group = parser.add_argument_group('model', 'model configuration')
    
    group.add_argument('--pretrained-bert', action='store_true',
                       help='use a pretrained bert-large-uncased model instead'
                       'of initializing from scratch. See '
                       '--tokenizer-model-type to specify which pretrained '
                       'BERT model to use')
    group.add_argument('--intermediate-size', type=int, default=None,
                       help='transformer embedding dimension for FFN'
                       'set to 4*`--hidden-size` if it is None')
    group.add_argument('--layernorm-epsilon', type=float, default=1e-5,
                       help='layer norm epsilon')
    group.add_argument('--deep-init', action='store_true',
                       help='initialize bert model similar to gpt2 model.'
                       'scales initialization of projection layers by a '
                       'factor of 1/sqrt(2N). Necessary to train bert '
                       'models larger than BERT-Large.')
    group.add_argument('--vocab-size', type=int, default=None,
                       help='vocabulary size to use for non-character-level '
                       'tokenization. This value will only be used when '
                       'creating a tokenizer')


    return parser


def add_fp16_config_args(parser):
    """Mixed precision arguments."""

    group = parser.add_argument_group('fp16', 'fp16 configurations')

    group.add_argument('--fp32-embedding', action='store_true',
                       help='embedding in fp32')
    group.add_argument('--fp32-layernorm', action='store_true',
                       help='layer norm in fp32')
    group.add_argument('--fp32-tokentypes', action='store_true',
                       help='embedding token types in fp32')
    group.add_argument('--fp32-allreduce', action='store_true',
                       help='all-reduce in fp32')

    return parser


def add_training_args_(parser):
    """Training arguments."""

    group = parser.add_argument_group('train', 'training configurations')

    # Batch prodecuer arguments
    group.add_argument('--reset-position-ids', action='store_true',
                       help='Reset posistion ids after end-of-document token.')
    group.add_argument('--reset-attention-mask', action='store_true',
                       help='Reset self attention maske after '
                       'end-of-document token.')
    group.add_argument('--eod-mask-loss', action='store_true',
                       help='Mask loss for the end of document tokens')

    # Learning rate.

    # autoresume
    group.add_argument('--adlr-autoresume', action='store_true',
                       help='enable autoresume on adlr cluster.')
    group.add_argument('--adlr-autoresume-interval', type=int, default=1000,
                       help='intervals over which check for autoresume'
                       'termination signal')

    return parser


def add_evaluation_args(parser):
    """Evaluation arguments."""

    group = parser.add_argument_group('validation', 'validation configurations')

    group.add_argument('--eval-batch-size', type=int, default=None,
                       help='Data Loader batch size for evaluation datasets.'
                       'Defaults to `--batch-size`')
    group.add_argument('--eval-seq-length', type=int, default=None,
                       help='Maximum sequence length to process for '
                       'evaluation. Defaults to `--seq-length`')
    group.add_argument('--eval-max-preds-per-seq', type=int, default=None,
                       help='Maximum number of predictions to use for '
                       'evaluation. Defaults to '
                       'math.ceil(`--eval-seq-length`*.15/10)*10')
    group.add_argument('--overlapping-eval', type=int, default=32,
                       help='sliding window for overlapping eval ')
    group.add_argument('--cloze-eval', action='store_true',
                       help='Evaluation dataset from `--valid-data` is a cloze task')
    group.add_argument('--strict-lambada', action='store_true',
                       help='use more difficult formulation of lambada')
    group.add_argument('--eval-hf', action='store_true',
                       help='perform evaluation with huggingface openai model.'
                       'use `--load` to specify weights path to be loaded')
    group.add_argument('--load-openai', action='store_true',
                       help='load openai weights into our model. Use `--load` '
                       'to specify weights path to be loaded')

    return parser

def add_text_generate_args(parser):
    """Text generate arguments."""

    group = parser.add_argument_group('Text generation', 'configurations')
    group.add_argument("--temperature", type=float, default=1.0)
    group.add_argument("--greedy", action='store_true', default=False)
    group.add_argument("--top_p", type=float, default=0.0)
    group.add_argument("--top_k", type=int, default=0)
    group.add_argument("--out-seq-length", type=int, default=1024)
    group.add_argument("--sample-input-file", type=str, default="",
                      help='get input from file instead of interactive mode, '
                           'each line is an input' )
    group.add_argument("--sample-output-file", type=str, default="",
                      help='output file got from --sample-input-file')
    group.add_argument("--num-samples", type=int, default=0,
                       help='number of samples to generate unconditionally, '
                       'defaults to 0 and interactive conditional sampling')
    group.add_argument("--genfile", type=str,
                       help='output file when generating unconditionally')
    group.add_argument("--recompute", action='store_true',
                       help='during generation recompute all attention '
                       'instead of using previously computed keys/values.')
    return parser


def add_data_args_(parser):
    """Train/valid/test data arguments."""

    group = parser.add_argument_group('data', 'data configurations')

    group.add_argument('--shuffle', action='store_true',
                       help='Shuffle data. Shuffling is deterministic '
                       'based on seed and current epoch.')
    group.add_argument('--data-loader', type=str, default=None,
                       choices=['raw', 'lazy', 'tfrecords', 'numpy', 'binary'],
                       help='Which data loader to use. Default varies by model.')

    group.add_argument('--train-data', nargs='+', default=None,
                       help='Whitespace separated paths or corpora names '
                       'for training.')
    group.add_argument('--valid-data', nargs='*', default=None,
                       help='path(s) to the validation data.')
    group.add_argument('--test-data', nargs='*', default=None,
                       help='path(s) to the testing data.')

    group.add_argument('--max-preds-per-seq', type=int, default=None,
                       help='Maximum number of predictions to use per sequence.'
                       'Defaults to math.ceil(`--seq-length`*.15/10)*10.'
                       'MUST BE SPECIFIED IF `--data-loader tfrecords`.')

    # arguments for binary data loader
    parser.add_argument('--data-impl', type=str, default='infer',
                        help='implementation of indexed datasets',
                        choices=['lazy', 'cached', 'mmap', 'infer'])
    parser.add_argument('--max-num-samples', type=int, default=None,
                        help='Maximum number of samples to plan for, defaults to total iters * batch-size.')
    parser.add_argument('--data-epochs', type=int, default=None,
                        help='Number of epochs to plan for, defaults to using --max-num-samples')

    # arguments for numpy data loader
    group.add_argument('--input-data-sizes-file', type=str, default='sizes.txt',
                       help='the filename containing all the shards sizes for numpy data loader')

    # arguments for raw/tfrecords data loader
    group.add_argument('--delim', default=',',
                       help='delimiter used to parse csv data files')
    group.add_argument('--text-key', default='sentence',
                       help='key to use to extract text from json/csv')
    group.add_argument('--eval-text-key', default=None,
                       help='key to use to extract text from '
                       'json/csv evaluation datasets')
    group.add_argument('--loose-json', action='store_true',
                       help='Use loose json (one json-formatted string per '
                       'newline), instead of tight json (data file is one '
                       'json string)')
    group.add_argument('--presplit-sentences', action='store_true',
                       help='Dataset content consists of documents where '
                       'each document consists of newline separated sentences')

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
                       default='BertWordPieceLowerCase',
                       choices=['CharacterLevelTokenizer',
                                'SentencePieceTokenizer',
                                'BertWordPieceLowerCase',
                                'GPT2BPETokenizer'],
                       help='what type of tokenizer to use')
    group.add_argument("--cache-dir", default=None, type=str,
                       help="Where to store pre-trained BERT downloads")

    return parser


def get_args_(extra_args_provider=None):
    """Parse all the args."""

    parser = argparse.ArgumentParser(description='Megatron-LM Arguments')

    parser = add_network_size_args(parser)
    parser = add_regularization_args(parser)
    parser = add_training_args(parser)
    parser = add_initialization_args(parser)
    parser = add_learning_rate_args(parser)
    parser = add_checkpointing_args(parser)
    parser = add_mixed_precision_args(parser)
    parser = add_distributed_args(parser)
    parser = add_validation_args(parser)
    parser = add_data_args(parser)

    #parser.print_help()
    #exit()

    parser = add_model_config_args(parser)
    parser = add_fp16_config_args(parser)
    parser = add_training_args_(parser)
    parser = add_evaluation_args(parser)
    parser = add_text_generate_args(parser)
    parser = add_data_args_(parser)
    if extra_args_provider is not None:
        parser = extra_args_provider(parser)


    args = parser.parse_args()

    # Checks.
    if args.save is not None:
        assert args.save_interval is not None, \
            'expected \'--save-interval\' in the input arguments.'

    
    if not args.train_data and not args.data_path:
        print('WARNING: No training data specified')

    args.cuda = torch.cuda.is_available()

    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))

    if os.getenv('OMPI_COMM_WORLD_LOCAL_RANK'):
        # We are using (OpenMPI) mpirun for launching distributed data parallel processes
        local_rank = int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK'))
        local_size = int(os.getenv('OMPI_COMM_WORLD_LOCAL_SIZE'))

        # Possibly running with Slurm
        num_nodes = int(os.getenv('SLURM_JOB_NUM_NODES', '1'))
        nodeid = int(os.getenv('SLURM_NODEID', '0'))

        args.local_rank = local_rank
        args.rank = nodeid*local_size + local_rank
        args.world_size = num_nodes*local_size

    args.model_parallel_size = min(args.model_parallel_size, args.world_size)
    if args.rank == 0:
        print('using world size: {} and model-parallel size: {} '.format(
            args.world_size, args.model_parallel_size))

    args.dynamic_loss_scale = False
    if args.loss_scale is None:
        args.dynamic_loss_scale = True
        if args.rank == 0:
            print(' > using dynamic loss scaling')

    # The args fp32_* or fp16_* meant to be active when the
    # args fp16 is set. So the default behaviour should all
    # be false.
    if not args.fp16:
        args.fp32_embedding = False
        args.fp32_tokentypes = False
        args.fp32_layernorm = False

    return args
