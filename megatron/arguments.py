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

"""Megatron arguments."""

import argparse
import os

import torch
from megatron import fused_kernels

def parse_args(extra_args_provider=None, defaults={},
               ignore_unknown_args=False):
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description='Megatron-LM Arguments',
                                     allow_abbrev=False)

    # Standard arguments.
    parser = _add_network_size_args(parser)
    parser = _add_regularization_args(parser)
    parser = _add_training_args(parser)
    parser = _add_initialization_args(parser)
    parser = _add_learning_rate_args(parser)
    parser = _add_checkpointing_args(parser)
    parser = _add_mixed_precision_args(parser)
    parser = _add_distributed_args(parser)
    parser = _add_validation_args(parser)
    parser = _add_data_args(parser)
    parser = _add_autoresume_args(parser)
    parser = _add_realm_args(parser)

    # Custom arguments.
    if extra_args_provider is not None:
        parser = extra_args_provider(parser)

    # Parse.
    if ignore_unknown_args:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()

    # Distributed args.
    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))
    args.model_parallel_size = min(args.model_parallel_size, args.world_size)
    if args.rank == 0:
        print('using world size: {} and model-parallel size: {} '.format(
            args.world_size, args.model_parallel_size))

    # Fp16 loss scaling.
    args.dynamic_loss_scale = False
    if args.loss_scale is None:
        args.dynamic_loss_scale = True

    # Parameters dtype.
    args.params_dtype = torch.float
    if args.fp16:
        args.params_dtype = torch.half
    if args.rank == 0:
        print('using {} for parameters ...'.format(args.params_dtype),
              flush=True)


    # Set input defaults.
    for key in defaults:
        # For default to be valid, it should not be provided in the
        # arguments that are passed to the program. We check this by
        # ensuring the arg is set to None.
        if getattr(args, key) is not None:
            if args.rank == 0:
                print('WARNING: overriding default arguments for {key}:{v} \
                       with {key}:{v2}'.format(key=key, v=defaults[key],
                                               v2=getattr(args, key)),
                                               flush=True)
        else:
            setattr(args, key, defaults[key])

    # Check required arguments.
    required_args = ['num_layers', 'hidden_size', 'num_attention_heads',
                     'max_position_embeddings']
    for req_arg in required_args: 
        _check_arg_is_not_none(args, req_arg)

    # Checks.
    assert args.hidden_size % args.num_attention_heads == 0
    if args.seq_length is not None:
        assert args.max_position_embeddings >= args.seq_length
    if args.lr is not None:
        assert args.min_lr <= args.lr
    if args.save is not None:
        assert args.save_interval is not None
    # Parameters sharing does not work with torch DDP.
    if (args.num_unique_layers is not None) and (args.num_layers is not None):
        assert args.num_unique_layers <= args.num_layers
        assert args.num_layers % args.num_unique_layers == 0, \
            'num-layers should be divisible by num-unique-layers.'
        if args.num_unique_layers < args.num_layers:
            assert args.DDP_impl == 'local', \
                'torch-DDP does not work with parameters sharing.'
    # Mixed precision checks.
    if args.fp16_lm_cross_entropy:
        assert args.fp16, 'lm cross entropy in fp16 only support in fp16 mode.'
    # Activation checkpointing.
    if args.distribute_checkpointed_activations:
        assert args.checkpoint_activations, \
            'for distribute-checkpointed-activations to work you '\
            'need to enable checkpoint-activations'

    # load scaled_upper_triang_masked_softmax_fusion kernel
    if args.scaled_upper_triang_masked_softmax_fusion:
        fused_kernels.load_scaled_upper_triang_masked_softmax_fusion_kernel()

    # load scaled_masked_softmax_fusion kernel
    if args.scaled_masked_softmax_fusion:
        fused_kernels.load_scaled_masked_softmax_fusion_kernel()

    _print_args(args)
    return args


def _print_args(args):
    """Print arguments."""
    if args.rank == 0:
        print('-------------------- arguments --------------------', flush=True)
        str_list = []
        for arg in vars(args):
            dots = '.' * (32 - len(arg))
            str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))
        for arg in sorted(str_list, key=lambda x: x.lower()):
            print(arg, flush=True)
        print('---------------- end of arguments ----------------', flush=True)


def _check_arg_is_not_none(args, arg):
    assert getattr(args, arg) is not None, '{} argument is None'.format(arg)


def _add_network_size_args(parser):
    group = parser.add_argument_group(title='network size')

    group.add_argument('--num-layers', type=int, default=None,
                       help='Number of transformer layers.')
    group.add_argument('--num-unique-layers', type=int, default=None,
                       help='Number of unique transformer layers. '
                       '`num-layers` should be divisible by this value.')
    group.add_argument('--param-sharing-style', default='grouped',
                       choices=['grouped', 'spaced'],
                       help='Ordering of the shared parameters. For example, '
                       'for a `num-layers`=4 and `--num-unique-layers`=2, '
                       'we will have the following ordering for two unique '
                       'layers 1 and 2: '
                       '    grouped: [1, 2, 1, 2] and spaced: [1, 1, 2, 2].')
    group.add_argument('--hidden-size', type=int, default=None,
                       help='Tansformer hidden size.')
    group.add_argument('--num-attention-heads', type=int, default=None,
                       help='Number of transformer attention heads.')
    group.add_argument('--max-position-embeddings', type=int, default=None,
                       help='Maximum number of position embeddings to use. '
                       'This is the size of position embedding.')
    group.add_argument('--make-vocab-size-divisible-by', type=int, default=128,
                       help='Pad the vocab size to be divisible by this value.'
                       'This is added for computational efficieny reasons.')
    group.add_argument('--layernorm-epsilon', type=float, default=1e-5,
                       help='Layer norm epsilon.')
    group.add_argument('--apply-residual-connection-post-layernorm',
                       action='store_true',
                       help='If set, use original BERT residula connection '
                       'ordering.')
    group.add_argument('--openai-gelu', action='store_true',
                       help='Use OpenAIs GeLU implementation. This option'
                       'should not be used unless for backward compatibility'
                       'reasons.')
    group.add_argument('--onnx-safe', type=bool, required=False,
                       help='Use workarounds for known problems with Torch ONNX exporter')

    return parser


def _add_regularization_args(parser):
    group = parser.add_argument_group(title='regularization')

    group.add_argument('--attention-dropout', type=float, default=0.1,
                       help='Post attention dropout ptobability.')
    group.add_argument('--hidden-dropout', type=float, default=0.1,
                       help='Dropout probability for hidden state transformer.')
    group.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay coefficient for L2 regularization.')
    group.add_argument('--clip-grad', type=float, default=1.0,
                       help='Gradient clipping based on global L2 norm.')
    group.add_argument('--adam-beta1', type=float, default=0.9,
                       help='First coefficient for computing running averages of'
                       'gradient and its square')
    group.add_argument('--adam-beta2', type=float, default=0.999,
                       help='Second coefficient for computing running averages of'
                       'gradient and its square')
    group.add_argument('--adam-eps', type=float, default=1e-08,
                       help='Term added to the denominator to improve'
                       'numerical stability')

    return parser


def _add_training_args(parser):
    group = parser.add_argument_group(title='training')

    group.add_argument('--batch-size', type=int, default=None,
                       help='Batch size per model instance (local batch size). '
                       'Global batch size is local batch size times data '
                       'parallel size.')
    group.add_argument('--checkpoint-activations', action='store_true',
                       help='Checkpoint activation to allow for training '
                       'with larger models, sequences, and batch sizes.')
    group.add_argument('--distribute-checkpointed-activations',
                       action='store_true',
                       help='If set, distribute checkpointed activations '
                       'across model parallel group.')
    group.add_argument('--checkpoint-num-layers', type=int, default=1,
                       help='chunk size (number of layers) for checkpointing.')
    group.add_argument('--train-iters', type=int, default=None,
                       help='Total number of iterations to train over all '
                       'training runs.')
    group.add_argument('--log-interval', type=int, default=100,
                       help='Report loss and timing interval.')
    group.add_argument('--exit-interval', type=int, default=None,
                       help='Exit the program after the iteration is divisible '
                       'by this value.')
    group.add_argument('--tensorboard-dir', type=str, default=None,
                       help='Write TensorBoard logs to this directory.')
    group.add_argument('--scaled-upper-triang-masked-softmax-fusion',
                       action='store_true',
                       help='Enable fusion of query_key_value_scaling '
                       'time (upper diagonal) masking and softmax.')
    group.add_argument('--scaled-masked-softmax-fusion',
                       action='store_true',
                       help='Enable fusion of query_key_value_scaling '
                       'general masking and softmax.')
    group.add_argument('--bias-gelu-fusion', action='store_true',
                        help='Enable bias and gelu fusion.')
    group.add_argument('--bias-dropout-fusion', action='store_true',
                       help='Enable bias and dropout fusion.')

    return parser


def _add_initialization_args(parser):
    group = parser.add_argument_group(title='initialization')

    group.add_argument('--seed', type=int, default=1234,
                       help='Random seed used for python, numpy, '
                       'pytorch, and cuda.')
    group.add_argument('--init-method-std', type=float, default=0.02,
                       help='Standard deviation of the zero mean normal '
                       'distribution used for weight initialization.')

    return parser


def _add_learning_rate_args(parser):
    group = parser.add_argument_group(title='learning rate')

    group.add_argument('--lr', type=float, default=None,
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


def _add_checkpointing_args(parser):
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


def _add_mixed_precision_args(parser):
    group = parser.add_argument_group(title='mixed precision')

    group.add_argument('--fp16', action='store_true',
                       help='Run model in fp16 mode.')
    group.add_argument('--apply-query-key-layer-scaling', action='store_true',
                       help='Scale Q * K^T by 1 / layer-number. If this flag '
                       'is set, then it will automatically set '
                       'attention-softmax-in-fp32 to true')
    group.add_argument('--attention-softmax-in-fp32', action='store_true',
                       help='Run attention masking and softmax in fp32.')
    group.add_argument('--fp32-allreduce', action='store_true',
                       help='All-reduce in fp32')
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
    group.add_argument('--fp16-lm-cross-entropy', action='store_true',
                       help='Move the cross entropy unreduced loss calculation'
                       'for lm head to fp16.')


    return parser


def _add_distributed_args(parser):
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
    group.add_argument('--lazy-mpu-init', type=bool, required=False,
                       help='If set to True, initialize_megatron() skips DDP initialization'
                       ' and returns function to complete it instead.'
                       'Also turns on --use-cpu-initialization flag.'
                       'This is for external DDP manager.' )
    group.add_argument('--use-cpu-initialization', action='store_true',
                       help='If set, affine parallel weights initialization uses CPU' )
    return parser


def _add_validation_args(parser):
    group = parser.add_argument_group(title='validation')

    group.add_argument('--eval-iters', type=int, default=100,
                       help='Number of iterations to run for evaluation'
                       'validation/test for.')
    group.add_argument('--eval-interval', type=int, default=1000,
                       help='Interval between running evaluation on '
                       'validation set.')

    return parser


def _add_data_args(parser):
    group = parser.add_argument_group(title='data and dataloader')

    group.add_argument('--data-path', type=str, default=None,
                       help='Path to combined dataset to split.')
    group.add_argument('--split', type=str, default='969, 30, 1',
                       help='Comma-separated list of proportions for training,'
                       ' validation, and test split. For example the split '
                       '`90,5,5` will use 90% of data for training, 5% for '
                       'validation and 5% for test.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file. Will be copied to the model directory. If omitted and a vocab '
                            'file already exists in the model directory, that file will be used.')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file. Will be copied to the model directory. If omitted and a merge '
                            'file already exists in the model directory, that file will be used.')
    group.add_argument('--seq-length', type=int, default=None,
                       help="Maximum sequence length to process.")
    group.add_argument('--mask-prob', type=float, default=0.15,
                       help='Probability of replacing a token with mask.')
    group.add_argument('--short-seq-prob', type=float, default=0.1,
                       help='Probability of producing a short sequence.')
    group.add_argument('--mmap-warmup', action='store_true',
                       help='Warm up mmap files.')
    group.add_argument('--num-workers', type=int, default=2,
                       help="Dataloader number of workers.")
    group.add_argument('--tokenizer-type', type=str,
                       default=None,
                       choices=['BertWordPieceLowerCase',
                                'BertWordPieceCase',
                                'GPT2BPETokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument('--data-impl', type=str, default='infer',
                       choices=['lazy', 'cached', 'mmap', 'infer'],
                       help='Implementation of indexed datasets.')
    group.add_argument('--reset-position-ids', action='store_true',
                       help='Reset posistion ids after end-of-document token.')
    group.add_argument('--reset-attention-mask', action='store_true',
                       help='Reset self attention maske after '
                       'end-of-document token.')
    group.add_argument('--eod-mask-loss', action='store_true',
                       help='Mask loss for the end of document tokens.')

    return parser


def _add_autoresume_args(parser):
    group = parser.add_argument_group(title='autoresume')

    group.add_argument('--adlr-autoresume', action='store_true',
                       help='Enable autoresume on adlr cluster.')
    group.add_argument('--adlr-autoresume-interval', type=int, default=1000,
                       help='Intervals over which check for autoresume'
                       'termination signal')

    return parser


def _add_realm_args(parser):
    group = parser.add_argument_group(title='realm')

    # network size
    group.add_argument('--ict-head-size', type=int, default=None,
                       help='Size of block embeddings to be used in ICT and REALM (paper default: 128)')

    # checkpointing
    group.add_argument('--ict-load', type=str, default=None,
                       help='Directory containing an ICTBertModel checkpoint')
    group.add_argument('--bert-load', type=str, default=None,
                       help='Directory containing an BertModel checkpoint (needed to start ICT and REALM)')

    # data
    group.add_argument('--titles-data-path', type=str, default=None,
                       help='Path to titles dataset used for ICT')
    group.add_argument('--query-in-block-prob', type=float, default=0.1,
                       help='Probability of keeping query in block for ICT dataset')
    group.add_argument('--use-one-sent-docs', action='store_true',
                       help='Whether to use one sentence documents in ICT')

    # training
    group.add_argument('--report-topk-accuracies', nargs='+', default=[],
                       help="Which top-k accuracies to report (e.g. '1 5 20')")

    # faiss index
    group.add_argument('--faiss-use-gpu', action='store_true',
                       help='Whether create the FaissMIPSIndex on GPU')
    group.add_argument('--block-data-path', type=str, default=None,
                       help='Where to save/load BlockData to/from')

    # indexer
    group.add_argument('--indexer-batch-size', type=int, default=128,
                       help='How large of batches to use when doing indexing jobs')
    group.add_argument('--indexer-log-interval', type=int, default=1000,
                       help='After how many batches should the indexer report progress')
    return parser

