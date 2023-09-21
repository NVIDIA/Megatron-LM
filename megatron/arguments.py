# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Megatron arguments."""

import argparse
import dataclasses
import json
import os
import torch
import deepspeed
import types
from packaging import version

import torch.nn.functional as F
from megatron.global_vars import set_retro_args, get_retro_args
from tools.retro.utils import get_args_path as get_retro_args_path

from megatron.core.transformer import TransformerConfig

def parse_args(extra_args_provider=None, ignore_unknown_args=False):
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
    parser = _add_biencoder_args(parser)
    parser = _add_vision_args(parser)
    parser = _add_logging_args(parser)
    parser = _add_zero_args(parser)
    parser = _add_memoryopt_args(parser)
    parser = _add_activation_checkpoint_args(parser)
    parser = _add_distillation_args(parser)
    parser = _add_inference_args(parser)
    parser = _add_transformer_engine_args(parser)
    parser = _add_retro_args(parser)

    # Custom arguments.
    if extra_args_provider is not None:
        parser = extra_args_provider(parser)

    parser = deepspeed.add_config_arguments(parser)

    # Parse.
    if ignore_unknown_args:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()

    # Args from environment
    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))

    return args

def validate_args(args):
    # Tensor model parallel size.
    args.tensor_model_parallel_size = min(
        args.tensor_model_parallel_size, args.world_size)
    assert args.world_size % args.tensor_model_parallel_size == 0, 'world size'\
        ' ({}) is not divisible by tensor model parallel size ({})'.format(
            args.world_size, args.tensor_model_parallel_size)
    # Pipeline model parallel size.
    args.pipeline_model_parallel_size = min(
        args.pipeline_model_parallel_size,
        (args.world_size // args.tensor_model_parallel_size))
    args.transformer_pipeline_model_parallel_size = (
        args.pipeline_model_parallel_size - 1
        if args.standalone_embedding_stage else
        args.pipeline_model_parallel_size
    )
    # Checks.
    if args.no_pipeline_parallel:
        assert args.pipeline_model_parallel_size == 1, \
            "pipeline_model_parallel_size must be 1 if pipeline parallel is disabled"
        
    if args.ds_sequence_parallel_size > 1:
        assert version.parse(deepspeed.__version__) >= version.parse("0.10.2"), "sequence parallelism requires DeepSpeed version 0.10.2+"

    model_parallel_size = args.pipeline_model_parallel_size * \
                          args.tensor_model_parallel_size * \
                          args.ds_sequence_parallel_size
    assert args.world_size % model_parallel_size == 0, 'world size ({}) is not'\
        ' divisible by tensor parallel size ({}) times pipeline parallel ' \
        'size ({})'.format(args.world_size, args.tensor_model_parallel_size,
                           args.pipeline_model_parallel_size)
    args.data_parallel_size = args.world_size // model_parallel_size
    if args.rank == 0:
        print('using world size: {}, data-parallel-size: {}, '
              'sequence-parallel size: {}, '
              'tensor-model-parallel size: {}, '
              'pipeline-model-parallel size: {} '.format(
                  args.world_size, args.data_parallel_size,
                  args.ds_sequence_parallel_size,
                  args.tensor_model_parallel_size,
                  args.pipeline_model_parallel_size), flush=True)
    if args.pipeline_model_parallel_size > 1:
        if args.pipeline_model_parallel_split_rank is not None:
            assert args.pipeline_model_parallel_split_rank < \
                    args.pipeline_model_parallel_size, 'split rank needs'\
                    ' to be less than pipeline model parallel size ({})'.format(
                            args.pipeline_model_parallel_size)

    # Deprecated arguments
    assert args.batch_size is None, '--batch-size argument is no longer ' \
        'valid, use --micro-batch-size instead'
    del args.batch_size
    assert args.warmup is None, '--warmup argument is no longer valid, use ' \
        '--lr-warmup-fraction instead'
    del args.warmup
    assert args.model_parallel_size is None, '--model-parallel-size is no ' \
        'longer valid, use --tensor-model-parallel-size instead'
    del args.model_parallel_size

    # HACK: below is commented because DeepSpeed still relies on the old
    # activation checkpointing mechanism.
    # if args.checkpoint_activations:
    #     if args.rank == 0:
    #         print('--checkpoint-activations is no longer valid, use --recompute-activations, '
    #               'or, for more control, --recompute-granularity and --recompute-method.')
    #     exit()
    # del args.checkpoint_activations

    if args.recompute_activations:
        args.recompute_granularity = 'selective'
    del args.recompute_activations

    args.ds_pipeline_enabled = not args.no_pipeline_parallel
    # Batch size.
    assert args.micro_batch_size is not None
    assert args.micro_batch_size > 0
    if args.global_batch_size is None:
        args.global_batch_size = args.micro_batch_size * args.data_parallel_size
        if args.rank == 0:
            print('setting global batch size to {}'.format(
                args.global_batch_size), flush=True)
    assert args.global_batch_size > 0
    if args.num_layers_per_virtual_pipeline_stage is not None:
        assert args.pipeline_model_parallel_size > 2, \
            'pipeline-model-parallel size should be greater than 2 with ' \
            'interleaved schedule'
        assert args.num_layers % args.num_layers_per_virtual_pipeline_stage == 0, \
            'number of layers is not divisible by number of layers per virtual ' \
            'pipeline stage'
        args.virtual_pipeline_model_parallel_size = \
            (args.num_layers // args.transformer_pipeline_model_parallel_size) // \
            args.num_layers_per_virtual_pipeline_stage
    else:
        args.virtual_pipeline_model_parallel_size = None

    # Parameters dtype.
    args.params_dtype = torch.float
    if args.fp16:
        assert not args.bf16
        args.params_dtype = torch.half
    if args.bf16:
        assert not args.fp16
        args.params_dtype = torch.bfloat16
        # bfloat16 requires gradient accumulation and all-reduce to
        # be done in fp32.
        if not args.accumulate_allreduce_grads_in_fp32:
            args.accumulate_allreduce_grads_in_fp32 = True
            if args.rank == 0:
                print('accumulate and all-reduce gradients in fp32 for '
                      'bfloat16 data type.', flush=True)

    if args.rank == 0:
        print('using {} for parameters ...'.format(args.params_dtype),
              flush=True)

    # If we do accumulation and all-reduces in fp32, we need to have local DDP
    # and we should make sure use-contiguous-buffers-in-local-ddp is not off.
    if args.accumulate_allreduce_grads_in_fp32:
        assert args.DDP_impl == 'local'
        assert args.use_contiguous_buffers_in_local_ddp

    # If we use the distributed optimizer, we need to have local DDP
    # and we should make sure use-contiguous-buffers-in-local-ddp is on.
    if args.use_distributed_optimizer:
        assert args.DDP_impl == 'local'
        assert args.use_contiguous_buffers_in_local_ddp

    # For torch DDP, we do not use contiguous buffer
    # if args.DDP_impl == 'torch':
    if args.DDP_impl != 'local':
        args.use_contiguous_buffers_in_local_ddp = False

    if args.dataloader_type is None:
        args.dataloader_type = 'single'

    # Consumed tokens.
    args.consumed_train_samples = 0
    args.consumed_valid_samples = 0
    args.consumed_train_tokens = 0

    # Support for variable sequence lengths across batches/microbatches.
    # set it if the dataloader supports generation of variable sequence lengths
    # across batches/microbatches. Due to additional communication overhead
    # during pipeline parallelism, it should not be set if sequence length
    # is constant during training.
    args.variable_seq_lengths = False

    # Iteration-based training.
    if args.train_iters:
        # If we use iteration-based training, make sure the
        # sample-based options are off.
        assert args.train_samples is None, \
            'expected iteration-based training'
        assert args.lr_decay_samples is None, \
            'expected iteration-based learning rate decay'
        assert args.lr_warmup_samples == 0, \
            'expected iteration-based learning rate warmup'
        assert args.rampup_batch_size is None, \
            'expected no batch-size rampup for iteration-based training'
        if args.lr_warmup_fraction is not None:
            assert args.lr_warmup_iters == 0, \
                'can only specify one of lr-warmup-fraction and lr-warmup-iters'

    # Sample-based training.
    if args.train_samples:
        # If we use sample-based training, make sure the
        # iteration-based options are off.
        assert args.train_iters is None, \
            'expected sample-based training'
        assert args.lr_decay_iters is None, \
            'expected sample-based learning rate decay'
        assert args.lr_warmup_iters == 0, \
            'expected sample-based learnig rate warmup'
        if args.lr_warmup_fraction is not None:
            assert args.lr_warmup_samples == 0, \
                'can only specify one of lr-warmup-fraction ' \
                'and lr-warmup-samples'

    if args.num_layers is not None:
        assert args.encoder_num_layers is None, \
            'cannot have both num-layers and encoder-num-layers specified'
        args.encoder_num_layers = args.num_layers
    else:
        if not args.use_dataset_only:
            assert args.encoder_num_layers is not None, \
                'either num-layers or encoder-num-layers should be specified'
            args.num_layers = args.encoder_num_layers

    # Check required arguments.
    if not args.use_dataset_only:
        required_args = ['num_layers', 'hidden_size', 'num_attention_heads',
                         'max_position_embeddings']
        for req_arg in required_args:
            _check_arg_is_not_none(args, req_arg)

    # Checks.
    if not args.use_dataset_only:
        if args.ffn_hidden_size is None:
            if args.swiglu:
                # reduce the dimnesion for MLP since projections happens on
                # two linear layers. this keeps the number of paramters in
                # the same ballpark as the counterpart with 4*h size
                # we keep it a multiple of 64, which means the actual tensor size
                # will be a multiple of 64 / tp_size
                args.ffn_hidden_size = int((4 * args.hidden_size * 2 / 3) / 64) * 64
            else:
                args.ffn_hidden_size = 4 * args.hidden_size

        if args.kv_channels is None:
            assert args.hidden_size % args.num_attention_heads == 0
            args.kv_channels = args.hidden_size // args.num_attention_heads

    if args.seq_length is not None:
        assert args.encoder_seq_length is None
        args.encoder_seq_length = args.seq_length
    else:
        assert args.encoder_seq_length is not None
        args.seq_length = args.encoder_seq_length

    if not args.use_dataset_only:
        if args.seq_length is not None:
            assert args.max_position_embeddings >= args.seq_length
        if args.decoder_seq_length is not None:
            assert args.max_position_embeddings >= args.decoder_seq_length
    # When rotary position embeddings is used, set add_position_embedding
    # to false to turn off absolute position embedding.
    if args.use_rotary_position_embeddings:
        args.add_position_embedding = False
    if args.lr is not None:
        assert args.min_lr <= args.lr
    if args.save is not None:
        assert args.save_interval is not None
    # Mixed precision checks.
    if args.fp16_lm_cross_entropy:
        assert args.fp16, 'lm cross entropy in fp16 only support in fp16 mode.'
    if args.fp32_residual_connection:
        assert args.fp16 or args.bf16, \
            'residual connection in fp32 only supported when using fp16 or bf16.'

    if not args.use_dataset_only:
        if args.weight_decay_incr_style == 'constant':
            assert args.start_weight_decay is None
            assert args.end_weight_decay is None
            args.start_weight_decay = args.weight_decay
            args.end_weight_decay = args.weight_decay
        else:
            assert args.start_weight_decay is not None
            assert args.end_weight_decay is not None

    TORCH_MAJOR = int(torch.__version__.split('.')[0])
    TORCH_MINOR = int(torch.__version__.split('.')[1])
    # Persistent fused layer norm.
    if TORCH_MAJOR < 1 or (TORCH_MAJOR == 1 and TORCH_MINOR < 11):
        args.no_persist_layer_norm = True
        if args.rank == 0:
            print('Persistent fused layer norm kernel is supported from '
                  'pytorch v1.11 (nvidia pytorch container paired with v1.11). '
                  'Defaulting to no_persist_layer_norm=True')

    # Activation checkpointing.
    if args.distribute_checkpointed_activations:
        assert args.checkpoint_activations, \
            'for distribute-checkpointed-activations to work you '\
            'need to enable checkpoint-activations'

    # Activation recomputing.
    if args.distribute_saved_activations:
        assert args.tensor_model_parallel_size > 1, 'can distribute ' \
            'recomputed activations only across tensor model ' \
            'parallel groups'
        assert args.recompute_granularity == 'full', \
            'distributed recompute activations is only '\
            'application to full recompute granularity'
        assert args.recompute_method is not None, \
            'for distributed recompute activations to work you '\
            'need to use a recompute method '
        assert TORCH_MAJOR >= 1 and TORCH_MINOR >= 10, \
            'distributed recompute activations are supported for pytorch ' \
            'v1.10 and above (Nvidia Pytorch container >= 21.07). Current ' \
            'pytorch version is v%s.%s.' % (TORCH_MAJOR, TORCH_MINOR)

    # Tranformer-Engine/FP8 related checking
    if args.fp8_e4m3 or args.fp8_hybrid:
        assert args.transformer_impl == 'transformer_engine', \
            'transformer-engine required for fp8 training and inference'

    assert not (args.fp8_e4m3 and args.fp8_hybrid), \
        'cannot train with both fp8 e4m3 and hybrid formatting'

    if args.recompute_granularity == 'selective':
        assert args.recompute_method is None, \
            'recompute method is not yet supported for ' \
            'selective recomputing granularity'

    # disable sequence parallelism when tp=1
    # to avoid change in numerics when
    # sequence_parallelism is enabled.
    if args.tensor_model_parallel_size == 1:
        args.sequence_parallel = False

    # disable async_tensor_model_parallel_allreduce when
    # model parallel memory optimization is enabled
    if args.sequence_parallel:
        args.async_tensor_model_parallel_allreduce = False

    # TODO: currently DeepSpeed seems to be incompatible with
    # async_tensor_model_parallel_allreduce thus temporarily disabling it.
    # Need further investigation.
    if args.deepspeed:
        args.async_tensor_model_parallel_allreduce = False

    if not args.use_dataset_only:
        if os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1":
            if args.sequence_parallel:
                raise RuntimeError(
                    "Using sequence parallelism requires setting the environment variable "
                    "CUDA_DEVICE_MAX_CONNECTIONS to 1")
            if args.async_tensor_model_parallel_allreduce:
                raise RuntimeError(
                    "Using async gradient all reduce requires setting the environment "
                    "variable CUDA_DEVICE_MAX_CONNECTIONS to 1")

    # Disable bias gelu fusion if we are disabling bias altogether
    if not args.add_bias_linear:
        args.bias_gelu_fusion = False

    # Load retro args.
    if args.retro_workdir:
        retro_args_path = get_retro_args_path(args.retro_workdir)
        if os.path.exists(retro_args_path):
            with open(retro_args_path) as f:
                retro_args = types.SimpleNamespace(**json.load(f))
                retro_args.retro_return_doc_ids = args.retro_return_doc_ids
                retro_args.retro_gpt_retrieved_length = \
                    args.retro_num_retrieved_chunks * \
                    retro_args.retro_gpt_chunk_length
                set_retro_args(retro_args)

    args.curriculum_learning_legacy = False
    args.compression_training = False

    # FlashAttention
    args.use_flash_attn = args.use_flash_attn_v1 or args.use_flash_attn_triton or args.use_flash_attn_v2

    # AML
    if args.aml_data_download_path is not None:
        data_paths = []
        for path in args.data_path:
            data_paths.append(f"{args.aml_data_download_path}/{path}")
        args.data_path = data_paths

    # GQA
    if not args.use_dataset_only:
        if args.num_key_value_heads is None:
            args.num_key_value_heads = args.num_attention_heads
        assert args.num_attention_heads % args.num_key_value_heads == 0, \
            f"num_attention_heads must be divisible by num_key_value_heads (got `num_attention_heads`: {args.num_attention_heads} " \
            f"and `num_key_value_heads`: {args.num_key_value_heads})."
        if args.num_key_value_heads != args.num_attention_heads:
            # if GQA
            assert not args.mos, 'GQA currently does not support args.mos'
            assert not args.kd, 'GQA currently does not support args.kd'

    # Print arguments.
    _print_args("arguments", args)
    retro_args = get_retro_args()
    if retro_args and args != retro_args:
        _print_args("retro arguments", types.SimpleNamespace(**{k:v for k,v in vars(retro_args).items() if k.startswith("retro")}, rank=args.rank))

    return args


def _print_args(title, args):
    """Print arguments."""
    if args.rank == 0:
        print(f'------------------------ {title} ------------------------',
              flush=True)
        str_list = []
        for arg in vars(args):
            dots = '.' * (48 - len(arg))
            str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))
        for arg in sorted(str_list, key=lambda x: x.lower()):
            print(arg, flush=True)
        print(f'-------------------- end of {title} ---------------------',
              flush=True)


def _check_arg_is_not_none(args, arg):
    assert getattr(args, arg) is not None, '{} argument is None'.format(arg)

def core_transformer_config_from_args(args):

    # Translate args to core transformer configuration
    kw_args = {}
    for f in dataclasses.fields(TransformerConfig):
        if hasattr(args, f.name):
            kw_args[f.name] = getattr(args, f.name)
    kw_args['persist_layer_norm'] = not args.no_persist_layer_norm
    kw_args['layernorm_zero_centered_gamma'] = args.apply_layernorm_1p
    kw_args['deallocate_pipeline_outputs'] = True
    kw_args['pipeline_dtype'] = args.params_dtype
    kw_args['batch_p2p_comm'] = not args.overlap_p2p_comm
    if args.swiglu:
        kw_args['activation_func'] = F.silu
        kw_args['gated_linear_unit'] = True
        kw_args['bias_gelu_fusion'] = False
    if args.init_method_xavier_uniform:
        kw_args['init_method'] = torch.nn.init.xavier_uniform_
        kw_args['scaled_init_method'] = torch.nn.init.xavier_uniform_

    return TransformerConfig(**kw_args)

def _add_transformer_engine_args(parser):
    group = parser.add_argument_group(title='Transformer-Engine')

    group.add_argument('--fp8-e4m3', action='store_true',
                        help='E4M3 TransformerLayer', dest='fp8_e4m3')
    group.add_argument('--fp8-hybrid', action='store_true',
                        help='Hybrid FP8 TransformerLayer', dest='fp8_hybrid')
    group.add_argument('--no-fp8-wgrad', action='store_false',
                        help='Execute wgrad in higher precision even for FP8 runs', dest='fp8_wgrad')
    group.add_argument('--fp8-margin', type=int, default=0,
                        help='Scaling margin for fp8', dest='fp8_margin')
    group.add_argument('--fp8-interval', type=int, default=1,
                        help='Scaling update interval for fp8', dest='fp8_interval')
    group.add_argument('--transformer-impl', default='local',
                       choices=['local', 'transformer_engine'],
                       help='Which Transformer implementation to use.',
                       dest='transformer_impl')
    group.add_argument('--fp8-amax-history-len', type=int, default=1,
                        help='Number of steps for which amax history is recorded per tensor',
                        dest='fp8_amax_history_len')
    group.add_argument('--fp8-amax-compute-algo', default='most_recent',
                       choices=['most_recent', 'max'],
                       help='Algorithm for computing amax from history',
                       dest='fp8_amax_compute_algo')

    return parser

def _add_inference_args(parser):
    group = parser.add_argument_group(title='inference')

    group.add_argument('--inference-batch-times-seqlen-threshold',
                       type=int, default=512,
                       help='During inference, if batch-size times '
                       'sequence-length is smaller than this threshold '
                       'then we will not use pipelining, otherwise we will.')
    group.add_argument('--max-tokens-to-oom',
                       type=int, default=12000,
                       help='Maximum number of tokens during inference'
                       'tokens here is # in prompt + # to generate'
                       'Allows us to throw an error before OOM crashes server')
    group.add_argument('--output-bert-embeddings', action='store_true',
                       help='Output Bert embeddings (via mean pooling) from '
                       'model, rather than its binary head output or entire '
                       'hidden batch.')
    group.add_argument('--bert-embedder-type', default="megatron",
                       choices=["megatron", "huggingface"],
                       help='Select either Megatron or Huggingface as the '
                       'Bert embedder.')

    return parser


def _add_retro_args(parser):
    group = parser.add_argument_group(title='retro')

    group.add_argument('--retro-workdir', default=None,
                       help='Retro working directory, which contains the '
                       'preprocessed data for for pretraining. This directory '
                       'is built during preprocessing (see '
                       'tools/retro/README.md), and contains subdirectories '
                       'for the chunk database and pretraining neighbors.')
    group.add_argument('--retro-add-retriever',
                       action='store_true', default=False,
                       help='Add a retriever to the transformer, for use in '
                       'pretraining a Retro model.')
    group.add_argument('--retro-cyclic-train-iters', type=int, default=None,
                       help='Set number of training iterations for cyclic '
                       'Retro training.')
    group.add_argument('--retro-encoder-layers', type=int, default=2,
                       help='Number of layers to use for the retrieval '
                       'encoder.')
    group.add_argument('--retro-encoder-hidden-dropout',
                       type=float, default=0.1, help='Hidden dropout for '
                       'retrieval encoder.')
    group.add_argument('--retro-encoder-attention-dropout',
                       type=float, default=0.1, help='Attention dropout for '
                       'retrieval encoder.')
    group.add_argument("--retro-num-neighbors", type=int, default=2,
                       help='Number of neighbors to retrieve during '
                       'pretraining.')
    group.add_argument("--retro-num-retrieved-chunks", type=int, default=2,
                       help='Number of chunks to retrieve from the retrieval '
                       'database.')
    group.add_argument("--retro-return-doc-ids", action="store_true",
                       help="Turn this on when preprocessing retro data.")

    # Enforce argument naming convention.
    for action in group._group_actions:
        prefix = action.dest.split("_")[0]
        assert prefix == "retro", \
            "Retro args must be prefixed with '--retro-*', for consistent " \
            "styling. Please fix '%s'." % ", ".join(action.option_strings)

    return parser


def _add_network_size_args(parser):
    group = parser.add_argument_group(title='network size')

    group.add_argument('--num-layers', type=int, default=None,
                       help='Number of transformer layers.')
    group.add_argument('--encoder-num-layers', type=int, default=None,
                       help='Number of encoder transformer layers.')
    group.add_argument('--decoder-num-layers', type=int, default=None,
                       help='Number of decoder transformer layers.')
    group.add_argument('--num-experts', type=int, nargs='+', default=[1,],
                           help='number of experts list, MoE related.')
    group.add_argument('--mlp-type', type=str, default='standard',
                           help='Only applicable when num-experts > 1, accepts [standard, residual]')
    group.add_argument('--topk', type=int, default=1,
                           help='Sets the k in TopK gating for MoE layers')
    group.add_argument('--expert-interval', type=int, default=2,
                           help='Use experts in every "expert-interval" layers')
    group.add_argument('--hidden-size', type=int, default=None,
                       help='Tansformer hidden size.')
    group.add_argument('--ffn-hidden-size', type=int, default=None,
                       help='Transformer Feed-Forward Network hidden size. '
                       'This is set to 4*hidden-size if not provided')
    group.add_argument('--num-attention-heads', type=int, default=None,
                       help='Number of transformer attention heads.')
    group.add_argument('--num-key-value-heads', type=int, default=None,
                       help='Number of key_value heads that should be used to implement Grouped Query Attention.')
    group.add_argument('--kv-channels', type=int, default=None,
                       help='Projection weights dimension in multi-head '
                       'attention. This is set to '
                       '   args.hidden_size // args.num_attention_heads '
                       'if not provided.')
    group.add_argument('--max-position-embeddings', type=int, default=None,
                       help='Maximum number of position embeddings to use. '
                       'This is the size of position embedding.')
    group.add_argument('--use-rotary-position-embeddings', action='store_true',
                       help='Use rotary positional embeddings or not')
    group.add_argument('--rotary-percent', type=float, default=1.0,
                       help='Percent of rotary dimension to use, default 100%')
    group.add_argument('--no-position-embedding',
                       action='store_false',
                       help='Disable position embedding.',
                       dest='add_position_embedding')
    group.add_argument('--make-vocab-size-divisible-by', type=int, default=128,
                       help='Pad the vocab size to be divisible by this value.'
                       'This is added for computational efficieny reasons.')
    group.add_argument('--normalization', type=str, default='layernorm',
                       choices=['layernorm', 'rmsnorm'],
                       help='Options for layer normalization type:'
                            '  layernorm'
                            '  rmsnorm')
    group.add_argument('--layernorm-epsilon', type=float, default=1e-5,
                       help='Layer norm epsilon.')
    group.add_argument('--apply-layernorm-1p', action='store_true',
                       help='Adjust LayerNorm weights such that they are centered '
                       'around zero. This improves numerical stability.')
    group.add_argument('--apply-residual-connection-post-layernorm',
                       action='store_true',
                       help='If set, use original BERT residula connection '
                       'ordering.')
    group.add_argument('--openai-gelu', action='store_true',
                       help='Use OpenAIs GeLU implementation. This option'
                       'should not be used unless for backward compatibility'
                       'reasons.')
    group.add_argument('--squared-relu', action='store_true',
                       help='Use squared relu activation instead of default gelu')
    group.add_argument('--swiglu', action='store_true',
                       help='Use gated linear units and SiLU activation instead of default gelu')
    group.add_argument('--onnx-safe', type=bool, required=False,
                       help='Use workarounds for known problems with '
                       'Torch ONNX exporter')
    group.add_argument('--bert-no-binary-head', action='store_false',
                       help='Disable BERT binary head.',
                       dest='bert_binary_head')
    group.add_argument('--num-experts-switch', type=int, default=None,
                       help='Number of Experts in Switch Transformer (None means no Switch)')
    group.add_argument('--untie-embeddings-and-output-weights', action='store_true',
                       help='Untie embeddings and output weights.'),
    group.add_argument('--embedding-weights-in-fp32', action='store_true',
                       help='Cast word embedding weights to fp32 before embedding fwd.'),
    return parser


def _add_logging_args(parser):
    group = parser.add_argument_group(title='logging')

    group.add_argument('--log-params-norm', action='store_true',
                       help='If set, calculate and log parameters norm.')
    group.add_argument('--log-num-zeros-in-grad', action='store_true',
                       help='If set, calculate and log the number of zeros in gradient.')
    group.add_argument('--timing-log-level', type=int,
                       default=0, choices=range(0,3),
                       help='Granularity level to measure and report timing. '
                       '   0: report only iteration time and make sure timing '
                       '      does not introduce extra overhead.'
                       '   1: report timing for operations that are executed '
                       '      very limited times (basically once) during '
                       '      each iteration (such as gradient all-reduce) '
                       '   2: report timing for operations that migh be '
                       '      executed numerous times during each iteration. '
                       'Note that setting the level to 1 or 2 might '
                       'cause increase in iteration time.')
    group.add_argument('--no-barrier-with-level-1-timing', action='store_false',
                       help='If not set, use barrier with level 1 time '
                       'measurements. Note that this is up to the user '
                       'to make sure calling barrier with their timers '
                       'will not result in hangs. This can happen if for '
                       'example the user adds a level 1 timer that is not '
                       'called by all ranks.',
                       dest='barrier_with_L1_time')
    group.add_argument('--timing-log-option', type=str, default='minmax',
                       choices=['max', 'minmax', 'all'],
                       help='Options for logging timing:'
                       '  max: report the max timing across all ranks'
                       '  minmax: report min and max timings across all ranks'
                       '  all: report timings of all ranks.')
    group.add_argument('--tensorboard-log-interval', type=int, default=1,
                       help='Report to tensorboard interval.')
    group.add_argument('--tensorboard-queue-size', type=int, default=1000,
                       help='Size of the tensorboard queue for pending events '
                       'and summaries before one of the ‘add’ calls forces a '
                       'flush to disk.')
    group.add_argument('--log-timers-to-tensorboard', action='store_true',
                       help='If set, write timers to tensorboard.')
    group.add_argument('--log-batch-size-to-tensorboard', action='store_true',
                       help='If set, write batch-size to tensorboard.')
    group.add_argument('--no-log-learnig-rate-to-tensorboard',
                       action='store_false',
                       help='Disable learning rate logging to tensorboard.',
                       dest='log_learning_rate_to_tensorboard')
    group.add_argument('--no-log-loss-scale-to-tensorboard',
                       action='store_false',
                       help='Disable loss-scale logging to tensorboard.',
                       dest='log_loss_scale_to_tensorboard')
    group.add_argument('--log-validation-ppl-to-tensorboard',
                       action='store_true',
                       help='If set, write validation perplexity to '
                       'tensorboard.')
    group.add_argument('--log-optimizer-states-to-tensorboard',
                       action='store_true',
                       help='If set, write various optimizer states to '
                       'tensorboard. This feature may consume extra GPU memory.')
    group.add_argument('--log-memory-to-tensorboard',
                       action='store_true',
                       help='Enable memory logging to tensorboard.')
    group.add_argument('--log-world-size-to-tensorboard',
                       action='store_true',
                       help='Enable world size logging to tensorboard.')

    return parser


def _add_regularization_args(parser):
    group = parser.add_argument_group(title='regularization')

    group.add_argument('--attention-dropout', type=float, default=0.1,
                       help='Post attention dropout probability.')
    group.add_argument('--hidden-dropout', type=float, default=0.1,
                       help='Dropout probability for hidden state transformer.')
    group.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay coefficient for L2 regularization.')
    group.add_argument('--start-weight-decay', type=float,
                       help='Initial weight decay coefficient for L2 regularization.')
    group.add_argument('--end-weight-decay', type=float,
                       help='End of run weight decay coefficient for L2 regularization.')
    group.add_argument('--weight-decay-incr-style', type=str, default='constant',
                       choices=['constant', 'linear', 'cosine'],
                       help='Weight decay increment function.')
    group.add_argument('--clip-grad', type=float, default=1.0,
                       help='Gradient clipping based on global L2 norm.')
    group.add_argument('--adam-beta1', type=float, default=0.9,
                       help='First coefficient for computing running averages '
                       'of gradient and its square')
    group.add_argument('--adam-beta2', type=float, default=0.999,
                       help='Second coefficient for computing running averages '
                       'of gradient and its square')
    group.add_argument('--adam-eps', type=float, default=1e-08,
                       help='Term added to the denominator to improve'
                       'numerical stability')
    group.add_argument('--sgd-momentum', type=float, default=0.9,
                       help='Momentum factor for sgd')

    return parser


def _add_training_args(parser):
    group = parser.add_argument_group(title='training')

    group.add_argument('--micro-batch-size', type=int, default=None,
                       help='Batch size per model instance (local batch size). '
                       'Global batch size is local batch size times data '
                       'parallel size times number of micro batches.')
    group.add_argument('--batch-size', type=int, default=None,
                       help='Old batch size parameter, do not use. '
                       'Use --micro-batch-size instead')
    group.add_argument('--global-batch-size', type=int, default=None,
                       help='Training batch size. If set, it should be a '
                       'multiple of micro-batch-size times data-parallel-size. '
                       'If this value is None, then '
                       'use micro-batch-size * data-parallel-size as the '
                       'global batch size. This choice will result in 1 for '
                       'number of micro-batches.')
    group.add_argument('--rampup-batch-size', nargs='*', default=None,
                       help='Batch size ramp up with the following values:'
                       '  --rampup-batch-size <start batch size> '
                       '                      <batch size incerement> '
                       '                      <ramp-up samples> '
                       'For example:'
                       '   --rampup-batch-size 16 8 300000 \ '
                       '   --global-batch-size 1024'
                       'will start with global batch size 16 and over '
                       ' (1024 - 16) / 8 = 126 intervals will increase'
                       'the batch size linearly to 1024. In each interval'
                       'we will use approximately 300000 / 126 = 2380 samples.')
    group.add_argument('--recompute-activations', action='store_true',
                       help='recompute activation to allow for training '
                       'with larger models, sequences, and batch sizes.')
    group.add_argument('--recompute-granularity', type=str, default=None,
                       choices=['full', 'selective'],
                       help='Checkpoint activations to allow for training '
                       'with larger models, sequences, and batch sizes. '
                       'It is supported at two granularities 1) full: '
                       'whole transformer layer is recomputed, '
                       '2) selective: core attention part of the transformer '
                       'layer is recomputed.')
    group.add_argument('--distribute-saved-activations',
                       action='store_true',
                       help='If set, distribute recomputed activations '
                       'across model parallel group.')
    group.add_argument('--recompute-method', type=str, default=None,
                       choices=['uniform', 'block'],
                       help='1) uniform: uniformly divide the total number of '
                       'Transformer layers and recompute the input activation of '
                       'each divided chunk at specified granularity, '
                       '2) recompute the input activations of only a set number of '
                       'individual Transformer layers per pipeline stage and do the '
                       'rest without any recomputing at specified granularity'
                       'default) do not apply activations recompute to any layers')
    group.add_argument('--recompute-num-layers', type=int, default=1,
                       help='1) uniform: the number of Transformer layers in each '
                       'uniformly divided recompute unit, '
                       '2) block: the number of individual Transformer layers '
                       'to recompute within each pipeline stage.')

    # deprecated
    # HACK: added back arguments because DeepSpeed still relies on the old
    # activation checkpointing mechanism.
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
                       'training runs. Note that either train-iters or '
                       'train-samples should be provided.')
    group.add_argument('--train-samples', type=int, default=None,
                       help='Total number of samples to train over all '
                       'training runs. Note that either train-iters or '
                       'train-samples should be provided.')
    group.add_argument('--train-tokens', type=int, default=None,
                       help='Total number of tokens to train over all '
                       'training runs.')
    group.add_argument('--random-ltd',
                       action='store_true',
                       help='enable random layer token drop')    
    group.add_argument('--log-interval', type=int, default=100,
                       help='Report loss and timing interval.')
    group.add_argument('--exit-interval', type=int, default=None,
                       help='Exit the program after the iteration is divisible '
                       'by this value.')
    group.add_argument('--exit-duration-in-mins', type=int, default=None,
                       help='Exit the program after this many minutes.')
    group.add_argument('--exit-signal-handler', action='store_true',
                       help='Dynamically save the checkpoint and shutdown the '
                       'training if SIGTERM is received')
    group.add_argument('--tensorboard-dir', type=str, default=None,
                       help='Write TensorBoard logs to this directory.')
    group.add_argument('--no-masked-softmax-fusion',
                       action='store_false',
                       help='Disable fusion of query_key_value scaling, '
                       'masking, and softmax.',
                       dest='masked_softmax_fusion')
    group.add_argument('--no-bias-gelu-fusion', action='store_false',
                       help='Disable bias and gelu fusion.',
                       dest='bias_gelu_fusion')
    group.add_argument('--no-bias-dropout-fusion', action='store_false',
                       help='Disable bias and dropout fusion.',
                       dest='bias_dropout_fusion')
    group.add_argument('--disable-moe-token-dropping', action='store_false',
                       help='Disable MoE expert token dropping.',
                       dest='moe_token_dropping')
    group.add_argument('--moe-train-capacity-factor', type=float, default=1.0,
                       help='The capacity of the MoE expert at training time')
    group.add_argument('--moe-eval-capacity-factor', type=float, default=1.0,
                       help='The capacity of the MoE expert at eval time.')
    group.add_argument('--moe-min-capacity', type=int, default=4,
                       help='The minimum capacity per MoE expert regardless of the capacity_factor.')
    group.add_argument('--moe-loss-coeff', type=float, default=0.1,
                       help='Scaling coefficient for adding MoE loss to model loss')
    group.add_argument('--create-moe-param-group', action='store_true',
                       help='Create separate groups for MoE params.'
                       'This is necessary for techniques like ZeRO.')
    group.add_argument('--use-flash-attn', '--use-flash-attn-v1', dest='use_flash_attn_v1', action='store_true',
                       help='use first version FlashAttention implementation of attention. '
                       'https://arxiv.org/abs/2205.14135')
    group.add_argument('--use-flash-attn-v2', action='store_true',
                       help='use second version FlashAttention implementation of attention. '
                       'https://arxiv.org/abs/2307.08691')
    group.add_argument('--use-flash-attn-triton', action='store_true',
                       help='use FlashAttention implementation of attention using Triton.')
    group.add_argument('--disable-bias-linear', action='store_false',
                       help='Disable bias in the linear layers',
                       dest='add_bias_linear')
    group.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'sgd'],
                       help='Optimizer function')
    group.add_argument('--dataloader-type', type=str, default=None,
                       choices=['single', 'cyclic'],
                       help='Single pass vs multiple pass data loader')
    group.add_argument('--ds-inference', action='store_true',
                       help='DeepSpeed inference engine being used')
    group.add_argument('--cpu-optimizer', action='store_true',
                       help='Run optimizer on CPU')
    group.add_argument('--cpu_torch_adam', action='store_true',
                       help='Use Torch Adam as optimizer on CPU.')
    group.add_argument('--no-pipeline-parallel', action='store_true',
                       help='Disable pipeline parallelism')
    group.add_argument('--use-tutel', action='store_true',
                       help='Use Tutel optimization for MoE')
    group.add_argument('--inference', action='store_true',
                       help='Very basic inference mode: not allocating optim/lr - requires ZERO_STAGE=0')

    group.add_argument('--no-async-tensor-model-parallel-allreduce',
                       action='store_false',
                       help='Disable asynchronous execution of '
                       'tensor-model-parallel all-reduce with weight '
                       'gradient compuation of a column-linear layer.',
                       dest='async_tensor_model_parallel_allreduce')
    group.add_argument('--no-persist-layer-norm', action='store_true',
                       help='Disable using persistent fused layer norm kernel. '
                       'This kernel supports only a set of hidden sizes. Please '
                       'check persist_ln_hidden_sizes if your hidden '
                       'size is supported.')
    group.add_argument('--sequence-parallel', action='store_true',
                       help='Enable Megatron-LM\'s sequence parallel optimization.')
    group.add_argument('--ds-sequence-parallel-size', type=int, default=1,
                       help='Enable DeepSpeed\'s sequence parallel. Cannot be combined with "--sequence-parallel", which enables Megatron-LM\'s sequence parallel.')
    group.add_argument('--force-ds-sequence-parallel', action='store_true',
                       help='use DeepSpeed sequence parallelism regardless of sequence parallel size.')
    group.add_argument('--no-gradient-accumulation-fusion',
                       action='store_false',
                       help='Disable fusing gradient accumulation to weight '
                       'gradient computation of linear layers',
                       dest='gradient_accumulation_fusion')
    group.add_argument('--use-dataset-only', type=bool, required=False, default=False,
                       help='If set to True, only use the megatron dataset for external trainer ')
    return parser


def _add_initialization_args(parser):
    group = parser.add_argument_group(title='initialization')

    group.add_argument('--seed', type=int, default=1234,
                       help='Random seed used for python, numpy, '
                       'pytorch, and cuda.')
    group.add_argument('--data-parallel-random-init', action='store_true',
                       help='Enable random initialization of params '
                       'across data parallel ranks')
    group.add_argument('--init-method-std', type=float, default=0.02,
                       help='Standard deviation of the zero mean normal '
                       'distribution used for weight initialization.')
    group.add_argument('--init-method-xavier-uniform', action='store_true',
                       help='Enable Xavier uniform parameter initialization')

    return parser


def _add_learning_rate_args(parser):
    group = parser.add_argument_group(title='learning rate')

    group.add_argument('--lr', type=float, default=None,
                       help='Initial learning rate. Depending on decay style '
                       'and initial warmup, the learing rate at each '
                       'iteration would be different.')
    group.add_argument('--lr-decay-style', type=str, default='linear',
                       choices=['constant', 'linear', 'cosine', 'inverse-square-root'],
                       help='Learning rate decay function.')
    group.add_argument('--lr-decay-iters', type=int, default=None,
                       help='number of iterations to decay learning rate over,'
                       ' If None defaults to `--train-iters`')
    group.add_argument('--lr-decay-samples', type=int, default=None,
                       help='number of samples to decay learning rate over,'
                       ' If None defaults to `--train-samples`')
    group.add_argument('--lr-decay-tokens', type=int, default=None,
                       help='number of tokens to decay learning rate over,'
                       ' If not None will override iter/sample-based decay')
    group.add_argument('--lr-warmup-fraction', type=float, default=None,
                       help='fraction of lr-warmup-(iters/samples) to use '
                       'for warmup (as a float)')
    group.add_argument('--lr-warmup-iters', type=int, default=0,
                       help='number of iterations to linearly warmup '
                       'learning rate over.')
    group.add_argument('--lr-warmup-samples', type=int, default=0,
                       help='number of samples to linearly warmup '
                       'learning rate over.')
    group.add_argument('--lr-warmup-tokens', type=int, default=None,
                       help='number of tokens to linearly warmup '
                       'learning rate over.')
    group.add_argument('--warmup', type=int, default=None,
                       help='Old lr warmup argument, do not use. Use one of the'
                       '--lr-warmup-* arguments above')
    group.add_argument('--min-lr', type=float, default=0.0,
                       help='Minumum value for learning rate. The scheduler'
                       'clip values below this threshold.')
    group.add_argument('--override-opt_param-scheduler', action='store_true',
                       help='Reset the values of the scheduler (learning rate,'
                       'warmup iterations, minimum learning rate, maximum '
                       'number of iterations, and decay style from input '
                       'arguments and ignore values from checkpoints. Note'
                       'that all the above values will be reset.')
    group.add_argument('--use-checkpoint-opt_param-scheduler', action='store_true',
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
    group.add_argument('--no-save-optim', action='store_true', default=None,
                       help='Do not save current optimizer.')
    group.add_argument('--no-save-rng', action='store_true', default=None,
                       help='Do not save current rng state.')
    group.add_argument('--load', type=str, default=None,
                       help='Directory containing a model checkpoint.')
    group.add_argument('--no-load-optim', action='store_true', default=None,
                       help='Do not load optimizer when loading checkpoint.')
    group.add_argument('--no-load-rng', action='store_true', default=None,
                       help='Do not load rng state when loading checkpoint.')
    group.add_argument('--no-load-lr-state', action='store_true',
                       help='Do not load lr state when loading checkpoint.')   
    group.add_argument('--finetune', action='store_true',
                       help='Load model for finetuning. Do not load optimizer '
                       'or rng state from checkpoint and set iteration to 0. '
                       'Assumed when loading a release checkpoint.')
    group.add_argument('--no-initialization', action='store_false',
                       help='Do not perform initialization when building model, '
                       'can reduce startup time when definitely loading from a '
                       'checkpoint',
                       dest='perform_initialization')
    group.add_argument('--use-checkpoint-args', action='store_true',
                       help='Override any command line arguments with arguments '
                       'from the checkpoint')
    group.add_argument('--exit-on-missing-checkpoint', action='store_true',
                       help="If '--load' is set, but checkpoint is not found "
                       "(e.g., path typo), then exit instead of random "
                       "initialization.")

    return parser


def _add_mixed_precision_args(parser):
    group = parser.add_argument_group(title='mixed precision')

    group.add_argument('--fp16', action='store_true',
                       help='Run model in fp16 mode.')
    group.add_argument('--bf16', action='store_true',
                       help='Run model in bfloat16 mode.')
    group.add_argument('--loss-scale', type=float, default=None,
                       help='Static loss scaling, positive power of 2 '
                       'values can improve fp16 convergence. If None, dynamic'
                       'loss scaling is used.')
    group.add_argument('--initial-loss-scale', type=float, default=2**32,
                       help='Initial loss-scale for dynamic loss scaling.')
    group.add_argument('--min-loss-scale', type=float, default=1.0,
                       help='Minimum loss scale for dynamic loss scale.')
    group.add_argument('--loss-scale-window', type=float, default=1000,
                       help='Window over which to raise/lower dynamic scale.')
    group.add_argument('--hysteresis', type=int, default=2,
                       help='hysteresis for dynamic loss scaling')
    group.add_argument('--fp32-residual-connection', action='store_true',
                       help='Move residual connections to fp32.')
    group.add_argument('--no-query-key-layer-scaling', action='store_false',
                       help='Do not scale Q * K^T by 1 / layer-number.',
                       dest='apply_query_key_layer_scaling')
    group.add_argument('--attention-softmax-in-fp32', action='store_true',
                       help='Run attention masking and softmax in fp32. '
                       'This flag is ignored unless '
                       '--no-query-key-layer-scaling is specified.')
    group.add_argument('--accumulate-allreduce-grads-in-fp32',
                       action='store_true',
                       help='Gradient accumulation and all-reduce in fp32.')
    group.add_argument('--fp16-lm-cross-entropy', action='store_true',
                       help='Move the cross entropy unreduced loss calculation'
                       'for lm head to fp16.')

    return parser


def _add_distributed_args(parser):
    group = parser.add_argument_group(title='distributed')

    group.add_argument('--tensor-model-parallel-size', type=int, default=1,
                       help='Degree of tensor model parallelism.')
    group.add_argument('--enable-expert-tensor-parallelism', action='store_true',
                        default=False,
                        help="use tensor parallelism for expert layers in MoE")
    group.add_argument('--pipeline-model-parallel-size', type=int, default=1,
                       help='Degree of pipeline model parallelism.')
    group.add_argument('--pipeline-model-parallel-split-rank',
                       type=int, default=None,
                       help='Rank where encoder and decoder should be split.')
    group.add_argument('--moe-expert-parallel-size', type=int, default=1,
                       help='Degree of the MoE expert parallelism.')
    group.add_argument('--model-parallel-size', type=int, default=None,
                       help='Old model parallel argument, do not use. Use '
                       '--tensor-model-parallel-size instead.')
    group.add_argument('--num-layers-per-virtual-pipeline-stage', type=int, default=None,
                       help='Number of layers per virtual pipeline stage')
    group.add_argument('--overlap-p2p-communication',
                       action='store_true',
                       help='overlap pipeline parallel communication with forward and backward chunks',
                       dest='overlap_p2p_comm')
    group.add_argument('--distributed-backend', default='nccl',
                       choices=['nccl', 'gloo', 'ccl'],
                       help='Which backend to use for distributed training.')
    group.add_argument('--distributed-timeout-minutes', type=int, default=10,
                       help='Timeout minutes for torch.distributed.')
    group.add_argument('--DDP-impl', default='local',
                       choices=['local', 'torch', 'FSDP'],
                       help='which DistributedDataParallel implementation '
                       'to use.')
    group.add_argument('--no-contiguous-buffers-in-local-ddp',
                       action='store_false', help='If set, dont use '
                       'contiguous buffer in local DDP.',
                       dest='use_contiguous_buffers_in_local_ddp')
    group.add_argument('--no-scatter-gather-tensors-in-pipeline', action='store_false',
                       help='Use scatter/gather to optimize communication of tensors in pipeline',
                       dest='scatter_gather_tensors_in_pipeline')
    group.add_argument('--use-ring-exchange-p2p', action='store_true',
                       default=False, help='If set, use custom-built ring exchange '
                       'for p2p communications. Note that this option will require '
                       'a custom built image that support ring-exchange p2p.')
    group.add_argument('--local-rank', '--local_rank', type=int, default=None,
                       help='local rank passed from distributed launcher.')
    group.add_argument('--lazy-mpu-init', type=bool, required=False,
                       help='If set to True, initialize_megatron() '
                       'skips DDP initialization and returns function to '
                       'complete it instead.Also turns on '
                       '--use-cpu-initialization flag. This is for '
                       'external DDP manager.' )
    group.add_argument('--use-cpu-initialization', action='store_true',
                       default=None, help='If set, affine parallel weights '
                       'initialization uses CPU' )
    group.add_argument('--empty-unused-memory-level', default=0, type=int,
                       choices=[0, 1, 2],
                       help='Call torch.cuda.empty_cache() each iteration '
                       '(training and eval), to reduce fragmentation.'
                       '0=off, 1=moderate, 2=aggressive.')
    group.add_argument('--standalone-embedding-stage', action='store_true',
                       default=False, help='If set, *input* embedding layer '
                       'is placed on its own pipeline stage, without any '
                       'transformer layers. (For T5, this flag currently only '
                       'affects the encoder embedding.)')
    group.add_argument('--use-distributed-optimizer', action='store_true',
                       help='Use distributed optimizer.')

    return parser


def _add_validation_args(parser):
    group = parser.add_argument_group(title='validation')

    group.add_argument('--eval-iters', type=int, default=100,
                       help='Number of iterations to run for evaluation'
                       'validation/test for.')
    group.add_argument('--eval-interval', type=int, default=1000,
                       help='Interval between running evaluation on '
                       'validation set.')
    group.add_argument('--skip-train', action='store_true',
                       default=False, help='If set, bypass the training loop, '
                       'optionally do evaluation for validation/test, and exit.')

    return parser


def _add_data_args(parser):
    group = parser.add_argument_group(title='data and dataloader')

    group.add_argument('--aml-data-download-path', type=str, default=None,
                       help='Path to mounted input dataset')
    group.add_argument('--data-path', nargs='*', default=None,
                       help='Path to the training dataset. Accepted format:'
                       '1) a single data path, 2) multiple datasets in the'
                       'form: dataset1-weight dataset1-path dataset2-weight '
                       'dataset2-path ... It is used with --split when a '
                       'single dataset used for all three: train, valid '
                       'and test. It is exclusive to the other '
                       '--*-data-path args')
    group.add_argument('--split', type=str, default='969, 30, 1',
                       help='Comma-separated list of proportions for training,'
                       ' validation, and test split. For example the split '
                       '`90,5,5` will use 90%% of data for training, 5%% for '
                       'validation and 5%% for test.')
    group.add_argument('--train-data-path', nargs='*', default=None,
                       help='Path to the training dataset. Accepted format:'
                       '1) a single data path, 2) multiple datasets in the'
                       'form: dataset1-weight dataset1-path dataset2-weight '
                       'dataset2-path ...')
    group.add_argument('--valid-data-path', nargs='*', default=None,
                       help='Path to the validation dataset. Accepted format:'
                       '1) a single data path, 2) multiple datasets in the'
                       'form: dataset1-weight dataset1-path dataset2-weight '
                       'dataset2-path ...')
    group.add_argument('--test-data-path', nargs='*', default=None,
                       help='Path to the test dataset. Accepted format:'
                       '1) a single data path, 2) multiple datasets in the'
                       'form: dataset1-weight dataset1-path dataset2-weight '
                       'dataset2-path ...')
    group.add_argument('--data-cache-path', default=None,
                       help='Path to a directory to hold cached index files.')

    group.add_argument('--vocab-size', type=int, default=None,
                       help='Size of vocab before EOD or padding.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file.')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file.')
    group.add_argument('--vocab-extra-ids', type=int, default=0,
                       help='Number of additional vocabulary tokens. '
                            'They are used for span masking in the T5 model')
    group.add_argument('--seq-length', type=int, default=None,
                       help='Maximum sequence length to process.')
    group.add_argument('--encoder-seq-length', type=int, default=None,
                       help='Maximum encoder sequence length to process.'
                       'This should be exclusive of --seq-length')
    group.add_argument('--decoder-seq-length', type=int, default=None,
                       help="Maximum decoder sequence length to process.")
    group.add_argument('--retriever-seq-length', type=int, default=256,
                       help='Maximum sequence length for the biencoder model '
                       'for retriever')
    group.add_argument('--sample-rate', type=float, default=1.0,
                       help='sample rate for training data. Supposed to be 0 '
                            ' < sample_rate < 1')
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
                                'GPT2BPETokenizer',
                                'SentencePieceTokenizer',
                                'GPTSentencePieceTokenizer',
                                'NullTokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument('--tokenizer-model', type=str, default=None,
                       help='Sentencepiece tokenizer model.')
    group.add_argument('--data-impl', type=str, default='infer',
                       choices=['mmap', 'infer'],
                       help='Implementation of indexed datasets.')
    group.add_argument('--reset-position-ids', action='store_true',
                       help='Reset posistion ids after end-of-document token.')
    group.add_argument('--reset-attention-mask', action='store_true',
                       help='Reset self attention maske after '
                       'end-of-document token.')
    group.add_argument('--eod-mask-loss', action='store_true',
                       help='Mask loss for the end of document tokens.')
    group.add_argument('--train-data-exact-num-epochs', type=int, default=None,
                       help='When building the train dataset, force it to be '
                       'an exact number of epochs of the raw data')
    group.add_argument('--return-data-index', action='store_true',
                       help='Return the index of data sample.')
    group.add_argument('--data-efficiency-curriculum-learning', action='store_true',
                       help='Use DeepSpeed data efficiency library curriculum learning feature.')
    group.add_argument('--train-idx-path', type=str, default=None,
                       help='Force to use certain index file.')
    group.add_argument('--train-desc-path', type=str, default=None,
                       help='Force to use certain index file.')
    group.add_argument('--train-doc-idx-path', type=str, default=None,
                       help='Force to use certain index file.')
    group.add_argument('--train-sample-idx-path', type=str, default=None,
                       help='Force to use certain index file.')
    group.add_argument('--train-shuffle-idx-path', type=str, default=None,
                       help='Force to use certain index file.')
    return parser


def _add_autoresume_args(parser):
    group = parser.add_argument_group(title='autoresume')

    group.add_argument('--adlr-autoresume', action='store_true',
                       help='Enable autoresume on adlr cluster.')
    group.add_argument('--adlr-autoresume-interval', type=int, default=1000,
                       help='Intervals over which check for autoresume'
                       'termination signal')

    return parser


def _add_biencoder_args(parser):
    group = parser.add_argument_group(title='biencoder')

    # network size
    group.add_argument('--ict-head-size', type=int, default=None,
                       help='Size of block embeddings to be used in ICT and '
                        'REALM (paper default: 128)')
    group.add_argument('--biencoder-projection-dim', type=int, default=0,
                       help='Size of projection head used in biencoder (paper'
                        ' default: 128)')
    group.add_argument('--biencoder-shared-query-context-model', action='store_true',
                        help='Whether to share the parameters of the query '
                        'and context models or not')

    # checkpointing
    group.add_argument('--ict-load', type=str, default=None,
                       help='Directory containing an ICTBertModel checkpoint')
    group.add_argument('--bert-load', type=str, default=None,
                       help='Directory containing an BertModel checkpoint '
                       '(needed to start ICT and REALM)')

    # data
    group.add_argument('--titles-data-path', type=str, default=None,
                       help='Path to titles dataset used for ICT')
    group.add_argument('--query-in-block-prob', type=float, default=0.1,
                       help='Probability of keeping query in block for '
                       'ICT dataset')
    group.add_argument('--use-one-sent-docs', action='store_true',
                       help='Whether to use one sentence documents in ICT')
    group.add_argument('--evidence-data-path', type=str, default=None,
                       help='Path to Wikipedia Evidence frm DPR paper')

    # training
    group.add_argument('--retriever-report-topk-accuracies', nargs='+', type=int,
                        default=[], help="Which top-k accuracies to report "
                        "(e.g. '1 5 20')")
    group.add_argument('--retriever-score-scaling', action='store_true',
                       help='Whether to scale retriever scores by inverse '
                        'square root of hidden size')

    # faiss index
    group.add_argument('--block-data-path', type=str, default=None,
                       help='Where to save/load BlockData to/from')
    group.add_argument('--embedding-path', type=str, default=None,
                       help='Where to save/load Open-Retrieval Embedding'
                        ' data to/from')

    # indexer
    group.add_argument('--indexer-batch-size', type=int, default=128,
                       help='How large of batches to use when doing indexing '
                       'jobs')
    group.add_argument('--indexer-log-interval', type=int, default=1000,
                       help='After how many batches should the indexer '
                       'report progress')
    return parser


def _add_vision_args(parser):
    group = parser.add_argument_group(title="vision")

    # general vision arguements
    group.add_argument('--num-classes', type=int, default=1000,
                       help='num of classes in vision classificaiton task')
    group.add_argument('--img-h', type=int, default=224,
                       help='Image height for vision classification task')
    group.add_argument('--img-w', type=int, default=224,
                       help='Image height for vision classification task')
    group.add_argument('--num-channels', type=int, default=3,
                       help='Number of channels in input image data')
    group.add_argument('--patch-dim', type=int, default=16,
                       help='patch dimension')
    group.add_argument('--classes-fraction', type=float, default=1.0,
                       help='training with fraction of classes.')
    group.add_argument('--data-per-class-fraction', type=float, default=1.0,
                       help='training with fraction of data per class.')
    group.add_argument('--no-data-sharding', action='store_false',
                       help='Disable data sharding.',
                       dest='data_sharding')
    group.add_argument('--head-lr-mult', type=float, default=1.0,
                       help='learning rate multiplier for head during finetuning')

    # pretraining type and backbone selection`
    group.add_argument('--vision-pretraining', action='store_true',
                       help='flag to indicate vision pretraining')
    group.add_argument('--vision-pretraining-type', type=str, default='classify',
                       choices=['classify', 'inpaint', 'dino'],
                       help='pretraining objectives')
    group.add_argument('--vision-backbone-type', type=str, default='vit',
                       choices=['vit', 'mit', 'swin'],
                       help='backbone types types')
    group.add_argument('--swin-backbone-type', type=str, default='tiny',
                       choices=['tiny', 'base', 'h3'],
                       help='pretraining objectives')

    # inpainting arguments
    group.add_argument('--mask-type', type=str, default='random',
                       choices=['random', 'row'],
                       help='mask types')
    group.add_argument('--mask-factor', type=float, default=1.0,
                       help='mask size scaling parameter')

    # dino arguments
    group.add_argument('--iter-per-epoch', type=int, default=1250,
                       help='iterations per epoch')
    group.add_argument('--dino-local-img-size', type=int, default=96,
                       help='Image size for vision classification task')
    group.add_argument('--dino-local-crops-number', type=int, default=10,
                       help='Number of local crops')
    group.add_argument('--dino-head-hidden-size', type=int, default=2048,
                       help='Hidden dimension size in dino head')
    group.add_argument('--dino-bottleneck-size', type=int, default=256,
                       help='Bottle neck dimension in dino head ')
    group.add_argument('--dino-freeze-last-layer', type=float, default=1,
                       help='Freezing last layer weights')
    group.add_argument('--dino-norm-last-layer', action='store_true',
                       help='Disable Norm in last layer.')
    group.add_argument('--dino-warmup-teacher-temp', type=float, default=0.04,
                       help='warump teacher temperature')
    group.add_argument('--dino-teacher-temp', type=float, default=0.07,
                       help='teacher temperature')
    group.add_argument('--dino-warmup-teacher-temp-epochs', type=int, default=30,
                       help='warmup teacher temperaure epochs')

    return parser


def _add_zero_args(parser):
    """Text generate arguments."""

    group = parser.add_argument_group('ZeRO configurations', 'configurations')
    group.add_argument("--zero-stage", type=int, default=1.0)
    group.add_argument('--zero-reduce-scatter', action='store_true',
                       help='Use reduce scatter if specified')
    group.add_argument('--zero-contigious-gradients', action='store_true',
                       help='Use contigious memory optimizaiton if specified')
    group.add_argument("--zero-reduce-bucket-size", type=int, default=0.0)
    group.add_argument("--zero-allgather-bucket-size", type=int, default=0.0)
    group.add_argument('--remote-device', type=str, default='none', choices=['none', 'cpu', 'nvme'],
                      help='Remote device for ZeRO-3 initialized parameters.')
    group.add_argument('--use-pin-memory', action='store_true',
                     help='Use pinned CPU memory for ZeRO-3 initialized model parameters.')
    return parser

def _add_memoryopt_args(parser):
    """Memory optimization arguments."""

    group = parser.add_argument_group('Memory optimizations', 'configurations')
    group.add_argument("--scattered-embeddings", action='store_true',
                       help='Save memory by scattering embedding activations. '
                            'Introduces dropout differences across MP configurations.')
    group.add_argument("--split-transformers", action='store_true',
                       help='Save memory by splitting transformer layers into two parts, '
                       'allowing for more frequent activation checkpoint savings.')
    group.add_argument("--memory-centric-tiled-linear", action="store_true",
                       help='Save memory by tiling with deepspeed.zero.TiledLinear.')
    group.add_argument("--tile-factor", type=int, default=1,
                       help='Make all linear layers the same size of [hidden/tile_factor, hidden/tile_factor]. '
                            'Must be enabled with --memory-centric-tiled-linear. '
                            'Example A: if tile_factor=1, the qkv layer [hidden, 3* hidden] would be converted into [1,3] tiles of size [hidden,hidden]. '
                            'Example B: if tile_factor=2, the intermediate layer [4*hidden, hidden] will be converted into [8, 2] tiles of size [hidden/2, hidden/2]. '
                            'Default is 1.')

    return parser

def _add_activation_checkpoint_args(parser):
    group = parser.add_argument_group('Activation Checkpointing',
                                      'Checkpointing Configurations')
    group.add_argument('--deepspeed-activation-checkpointing', action='store_true',
                       help='uses activation checkpointing from deepspeed')
    group.add_argument('--partition-activations', action='store_true',
                       help='partition Activations across GPUs before checkpointing.')
    group.add_argument('--contigious-checkpointing', action='store_true',
                       help='Contigious memory checkpointing for activatoins.')
    group.add_argument('--checkpoint-in-cpu', action='store_true',
                       help='Move the activation checkpoints to CPU.')
    group.add_argument('--synchronize-each-layer', action='store_true',
                       help='does a synchronize at the beginning and end of each checkpointed layer.')
    group.add_argument('--profile-backward', action='store_true',
                       help='Enables backward pass profiling for checkpointed layers.')
    return parser


def _add_distillation_args(parser):
    group = parser.add_argument_group('Knowledge distillation',
                                      'Distillation Configurations')
    
    group.add_argument('--num-layers-teacher', type=int, default=None,
                       help='Number of the teacher transformer layers.')                  
    group.add_argument('--num-experts-teacher', type=int, nargs='+', default=[1,],
                        help='number of teacher experts list, MoE related.')
    group.add_argument('--hidden-size-teacher', type=int, default=None,
                       help='Tansformer teacher hidden size.')
    group.add_argument('--num-attention-heads-teacher', type=int, default=None,
                       help='Number of teacher transformer attention heads.') 

    group.add_argument('--mos', action='store_true',
                       help='Enable Mixture-of-Students via knolwedge distillation.')
    group.add_argument('--kd', action='store_true',
                       help='Enable knolwedge distillation.')
    group.add_argument('--kd-alpha-ce', default=1, type=float)
    group.add_argument('--kd-beta-ce', default=1, type=float)
    group.add_argument('--kd-temp', default=1.0, type=float)
    group.add_argument('--reset-iteration', action='store_true',
                    help='Reset the iteration count.')
    
    group.add_argument('--load-teacher', type=str, default=None,
                       help='Directory containing a teacher model checkpoint.')

    return parser
