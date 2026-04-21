# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Megatron arguments."""

import argparse
import dataclasses
import json
import os
import torch
import types

from itertools import chain, starmap
from types import SimpleNamespace
import yaml, re, os
from types import SimpleNamespace

import torch.nn.functional as F

from megatron.core.transformer import TransformerConfig, MLATransformerConfig
from megatron.core.utils import get_torch_version, is_torch_min_version

# Taken from https://stackoverflow.com/questions/65414773/parse-environment-variable-from-yaml-with-pyyaml
# Allows for yaml to use environment variables
env_pattern = re.compile(r".*?\${(.*?)}.*?")
def env_constructor(loader, node):
    value = loader.construct_scalar(node)
    for group in env_pattern.findall(value):
        assert os.environ.get(group) is not None, f"environment variable {group} in yaml not found"
        value = value.replace(f"${{{group}}}", os.environ.get(group))
    return value
yaml.add_implicit_resolver("!pathex", env_pattern)
yaml.add_constructor("!pathex", env_constructor)


str_dtype_to_torch = {
    "float32" : torch.float32,
    "float16" : torch.float16,
    "bfloat16" : torch.bfloat16
}

def validate_yaml(args, defaults={}):
    
    # This is for legacy script env var setting
    if type(args.data_path) is str:
        # If no white space its a single path
        split_data_path = args.data_path.split()
        if len(split_data_path) != 1:
            args.data_path = split_data_path

    # Tensor model parallel size.
    args.model_parallel.tensor_model_parallel_size = min(
        args.model_parallel.tensor_model_parallel_size, args.world_size)
    assert args.world_size % args.model_parallel.tensor_model_parallel_size == 0, 'world size'\
        ' ({}) is not divisible by tensor model parallel size ({})'.format(
            args.world_size, args.model_parallel.tensor_model_parallel_size)
    # Pipeline model parallel size.
    args.model_parallel.pipeline_model_parallel_size = min(
        args.model_parallel.pipeline_model_parallel_size,
        (args.world_size // args.model_parallel.tensor_model_parallel_size))
    args.model_parallel.transformer_pipeline_model_parallel_size = (
        args.model_parallel.pipeline_model_parallel_size - 1
        if args.account_for_embedding_in_pipeline_split else
        args.model_parallel.pipeline_model_parallel_size
    )
    # Checks.
    model_parallel_size = args.model_parallel.pipeline_model_parallel_size * \
                          args.model_parallel.tensor_model_parallel_size
    assert args.world_size % (model_parallel_size * args.model_parallel.context_parallel_size) == 0, \
        'world size ({}) is not divisible by tensor parallel size ({}) times ' \
        'pipeline parallel size ({}) times context parallel size ({})'.format(
        args.world_size, args.model_parallel.tensor_model_parallel_size,
        args.model_parallel.pipeline_model_parallel_size, args.model_parallel.context_parallel_size)
    
    # data_parallel_size is not in model parallel config
    args.data_parallel_size = args.world_size // (model_parallel_size * args.model_parallel.context_parallel_size)
    if args.rank == 0:
        print('using world size: {}, data-parallel size: {}, '
              'context-parallel size: {} '
              'tensor-model-parallel size: {}, '
              'pipeline-model-parallel size: {} '.format(
                  args.world_size, args.data_parallel_size,
                  args.model_parallel.context_parallel_size,
                  args.model_parallel.tensor_model_parallel_size,
                  args.model_parallel.pipeline_model_parallel_size), flush=True)
    if args.model_parallel.pipeline_model_parallel_size > 1:
        if args.model_parallel.pipeline_model_parallel_split_rank is not None:
            assert args.model_parallel.pipeline_model_parallel_split_rank < \
                    args.model_parallel.pipeline_model_parallel_size, 'split rank needs'\
                    ' to be less than pipeline model parallel size ({})'.format(
                            args.model_parallel.pipeline_model_parallel_size)

    if args.model_parallel.tp_comm_overlap:
        assert args.model_parallel.sequence_parallel == True, 'Tensor parallel communication/GEMM overlap can happen only when sequence parallelism is enabled'

    # Set input defaults.
    for key in defaults:
        # For default to be valid, it should not be provided in the
        # arguments that are passed to the program. We check this by
        # ensuring the arg is set to None.
        if getattr(args, key, None) is not None:
            if args.rank == 0:
                print('WARNING: overriding default arguments for {key}:{v} \
                       with {key}:{v2}'.format(key=key, v=defaults[key],
                                               v2=getattr(args, key)),
                                               flush=True)
        else:
            setattr(args, key, defaults[key])

    # Batch size.
    assert args.micro_batch_size is not None
    assert args.micro_batch_size > 0
    if args.global_batch_size is None:
        args.global_batch_size = args.micro_batch_size * args.data_parallel_size
        if args.rank == 0:
            print('setting global batch size to {}'.format(
                args.global_batch_size), flush=True)
    assert args.global_batch_size > 0

    # num_layers_per_virtual_pipeline_stage is not insde model parallel for checkpointing
    if args.num_layers_per_virtual_pipeline_stage is not None:
        assert args.model_parallel.pipeline_model_parallel_size > 2, \
            'pipeline-model-parallel size should be greater than 2 with ' \
            'interleaved schedule'
        assert args.language_model.num_layers % args.model_parallel.transformer_pipeline_model_parallel_size == 0, \
            'number of layers should be divisible by the pipeline parallel size'
        num_layers_per_pipeline_stage = args.language_model.num_layers // args.model_parallel.transformer_pipeline_model_parallel_size
        assert num_layers_per_pipeline_stage % args.num_layers_per_virtual_pipeline_stage == 0, \
            'number of layers per pipeline stage must be divisible number of layers per virtual pipeline stage'
        args.model_parallel.virtual_pipeline_model_parallel_size = num_layers_per_pipeline_stage // \
            args.num_layers_per_virtual_pipeline_stage
    else:
        args.model_parallel.virtual_pipeline_model_parallel_size = None
        # Overlap P2P communication is disabled if not using the interleaved schedule.
        args.model_parallel.overlap_p2p_comm = False
        if args.rank == 0:
            print('WARNING: Setting args.overlap_p2p_comm to False since non-interleaved '
                  'schedule does not support overlapping p2p communication')

    if args.overlap_param_gather:
        assert args.use_distributed_optimizer, \
            '--overlap-param-gather only supported with distributed optimizer'
        assert args.overlap_grad_reduce, \
            '--overlap-grad-reduce should be turned on when using --overlap-param-gather'

    # Parameters dtype.
    if args.model_parallel.fp16:
        assert not args.model_parallel.bf16
        args.model_parallel.params_dtype = torch.half
    if args.model_parallel.bf16:
        assert not args.model_parallel.fp16
        args.model_parallel.params_dtype = torch.bfloat16
        # bfloat16 requires gradient accumulation and all-reduce to
        # be done in fp32.
        if not args.accumulate_allreduce_grads_in_fp32:
            args.accumulate_allreduce_grads_in_fp32 = True
            if args.rank == 0:
                print('accumulate and all-reduce gradients in fp32 for '
                      'bfloat16 data type.', flush=True)

    if args.rank == 0:
        print('using {} for parameters ...'.format(args.model_parallel.params_dtype),
              flush=True)

    if args.dataloader_type is None:
        args.dataloader_type = 'single'

    # Consumed tokens.
    args.consumed_train_samples = 0
    args.consumed_valid_samples = 0

    # Support for variable sequence lengths across batches/microbatches.
    # set it if the dataloader supports generation of variable sequence lengths
    # across batches/microbatches. Due to additional communication overhead
    # during pipeline parallelism, it should not be set if sequence length
    # is constant during training.
    args.model_parallel.variable_seq_lengths = False

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

    # How to handle this better
    if args.language_model.num_layers is not None:
        assert args.encoder_num_layers is None, \
            'cannot have both num-layers and encoder-num-layers specified'
        args.encoder_num_layers = args.language_model.num_layers
    else:
        assert args.encoder_num_layers is not None, \
            'either num-layers or encoder-num-layers should be specified'
        args.language_model.num_layers = args.encoder_num_layers

    # Check required arguments.
    # removed max_position_embeddings from reqs
    required_args = ['num_layers', 'hidden_size', 'num_attention_heads']
    for req_arg in required_args:
        _check_arg_is_not_none(args.language_model, req_arg)

    # Checks.
    if args.language_model.ffn_hidden_size is None:
        if args.language_model.activation_func == "swiglu":
            # reduce the dimnesion for MLP since projections happens on
            # two linear layers. this keeps the number of paramters in
            # the same ballpark as the counterpart with 4*h size
            # we keep it a multiple of 64, which means the actual tensor size
            # will be a multiple of 64 / tp_size
            args.language_model.ffn_hidden_size = int((4 * args.language_model.hidden_size * 2 / 3) / 64) * 64
        else:
            args.language_model.ffn_hidden_size = 4 * args.language_model.hidden_size

    if args.language_model.kv_channels is None:
        assert args.language_model.hidden_size % args.language_model.num_attention_heads == 0
        args.language_model.kv_channels = args.language_model.hidden_size // args.language_model.num_attention_heads

    #TODO: Implement arguments for encoder-decoder
    if args.seq_length is not None:
        assert args.encoder_seq_length is None
        args.encoder_seq_length = args.seq_length
    else:
        assert args.encoder_seq_length is not None
        args.seq_length = args.encoder_seq_length

    if args.seq_length is not None:
        assert args.max_position_embeddings >= args.seq_length
    if args.decoder_seq_length is not None:
        assert args.max_position_embeddings >= args.decoder_seq_length
    if args.lr is not None:
        assert args.min_lr <= args.lr
    if args.save is not None:
        assert args.save_interval is not None
    # Mixed precision checks.
    if args.fp16_lm_cross_entropy:
        assert args.fp16, 'lm cross entropy in fp16 only support in fp16 mode.'
    if args.language_model.fp32_residual_connection:
        assert args.model_parallel.fp16 or args.model_parallel.bf16, \
            'residual connection in fp32 only supported when using fp16 or bf16.'

    if args.language_model.moe_grouped_gemm:
        assert args.model_parallel.bf16, 'Currently GroupedGEMM for MoE only supports bf16 dtype.'
        dc = torch.cuda.get_device_capability()
        assert dc[0] >= 8, "Unsupported compute capability for GroupedGEMM kernels."

    if args.weight_decay_incr_style == 'constant':
        assert args.start_weight_decay is None
        assert args.end_weight_decay is None
        args.start_weight_decay = args.weight_decay
        args.end_weight_decay = args.weight_decay
    else:
        assert args.start_weight_decay is not None
        assert args.end_weight_decay is not None

    # Persistent fused layer norm.
    if not is_torch_min_version("1.11.0a0"):
        args.language_model.persist_layer_norm = False
        if args.rank == 0:
            print('Persistent fused layer norm kernel is supported from '
                  'pytorch v1.11 (nvidia pytorch container paired with v1.11). '
                  'Defaulting to no_persist_layer_norm=True')

    # Activation recomputing.
    if args.language_model.distribute_saved_activations:
        assert args.model_parallel.tensor_model_parallel_size > 1, 'can distribute ' \
            'recomputed activations only across tensor model ' \
            'parallel groups'
        assert args.language_model.recompute_granularity == 'full', \
            'distributed recompute activations is only '\
            'application to full recompute granularity'
        assert args.language_model.recompute_method is not None, \
            'for distributed recompute activations to work you '\
            'need to use a recompute method '
        assert is_torch_min_version("1.10.0a0"), \
            'distributed recompute activations are supported for pytorch ' \
            'v1.10 and above (Nvidia Pytorch container >= 21.07). Current ' \
            f'pytorch version is v{get_torch_version()}.'

    if args.language_model.recompute_granularity == 'selective':
        assert args.language_model.recompute_method is None, \
            'recompute method is not yet supported for ' \
            'selective recomputing granularity'

    # disable sequence parallelism when tp=1
    # to avoid change in numerics when
    # sequence_parallelism is enabled.
    if args.model_parallel.tensor_model_parallel_size == 1:
        args.model_parallel.sequence_parallel = False

    # disable async_tensor_model_parallel_allreduce when
    # model parallel memory optimization is enabled
    if args.model_parallel.sequence_parallel:
        args.model_parallel.async_tensor_model_parallel_allreduce = False

    if os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1":
        if args.model_parallel.sequence_parallel:
            raise RuntimeError(
                "Using sequence parallelism requires setting the environment variable "
                "CUDA_DEVICE_MAX_CONNECTIONS to 1")
        if args.model_parallel.async_tensor_model_parallel_allreduce:
            raise RuntimeError(
                "Using async gradient all reduce requires setting the environment "
                "variable CUDA_DEVICE_MAX_CONNECTIONS to 1")

    # Retro checks.
    if getattr(args, 'retro_add_retriever', False):
        raise Exception("Retro untested for yaml args. See arguments.py.")

        # Sequence parallelism unsupported.
        assert not args.sequence_parallel, \
            "retro currently does not support sequence parallelism."

        # Pipeline parallelism unsupported.
        assert args.pipeline_model_parallel_size == 1, \
            "retro currently does not support pipeline parallelism."

    #TODO: Retro args loading not tested
    # Load retro args (used by both Retro & GPT).
    if getattr(args, 'retro_project_dir', None) is not None:
        raise Exception("Retro untested for yaml args. See arguments.py.")

    if args.language_model.rotary_interleaved and args.language_model.apply_rope_fusion:
        raise RuntimeError('--rotary-interleaved does not work with rope_fusion.')
    
    # MoE Spec check
    if args.language_model.num_moe_experts is not None:
        assert args.spec is None, "Model Spec must be None when using MoEs"
        if args.model_parallel.tensor_model_parallel_size > 1:
            assert args.model_parallel.sequence_parallel, \
                "When using MoE and tensor parallelism, sequence parallelism must be used."

    # Expert parallelism check
    if args.model_parallel.expert_model_parallel_size  > 1:
        assert args.language_model.num_moe_experts is not None, "num_experts must be non None to use expert model parallelism"
        assert args.language_model.num_moe_experts % args.model_parallel.expert_model_parallel_size == 0, \
            "Number of experts should be a multiple of expert model parallel_size."
        assert not args.model_parallel.fp16, \
            "Expert parallelism is not supported with fp16 training."

    # Print arguments.
    _print_args("arguments", args)

    #TODO: Added as much of the global initialization requires the model parallel arguments
    args = SimpleNamespace(**args.__dict__, **args.model_parallel.__dict__)
    args = SimpleNamespace(**args.__dict__, **args.language_model.__dict__)
    # For GPT Layer spec in pretrain_gpt
    args.num_experts = args.language_model.num_moe_experts

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

def core_config_from_args(args, dataclass=TransformerConfig):
    """Builds core config object from namespace args from given dataclass

    Raises exception if argument missing in args

    Args:
        args(SimpleNamespace, optional): Namespace to pull argument values from 
        dataclass (dataclass, optional): Core dataclass config to pull argument names from


    Returns:
        SimpleNamespace: The returned namespace to build core config from
    """
    kw_args = {}
    for f in dataclasses.fields(dataclass):
        if hasattr(args, f.name):
            kw_args[f.name] = getattr(args, f.name)
        else:
            raise Exception(f"Missing argument {f.name} for {str(dataclass)} config")
    return kw_args

def _check_arg_is_not_none(args, arg):
    assert getattr(args, arg) is not None, '{} argument is None'.format(arg)

def core_transformer_config_from_yaml(args, transfomer_key = "language_model"):    
    # Combine transfomer config with model parallel args
    args = SimpleNamespace(**vars(getattr(args, transfomer_key)), **vars(args.model_parallel))
    # Translate args to core transformer configuration
    kw_args = core_config_from_args(args, TransformerConfig)    
    
    # Hardcoded 
    kw_args['deallocate_pipeline_outputs'] = True
    kw_args['pipeline_dtype'] = kw_args['params_dtype']
    kw_args['batch_p2p_comm'] = not args.overlap_p2p_comm 
    
    assert args.activation_func in ["swiglu","squaredrelu","gelu"], f"{args.activation_func} is not a supported activation function"
    if args.activation_func == "swiglu":
        kw_args['activation_func'] = F.silu
        kw_args['gated_linear_unit'] = True
        kw_args['bias_activation_fusion'] = args.bias_swiglu_fusion
    elif args.activation_func == "squaredrelu":
        def squared_relu(x):
            return torch.pow(F.relu(x), 2)
        kw_args['activation_func'] = squared_relu
    elif args.activation_func == "gelu":
        kw_args['activation_func'] = F.gelu
        if args.add_bias_linear:
            kw_args['bias_activation_fusion'] = False
        else:
            kw_args['bias_activation_fusion'] = args.bias_activation_fusion
    
    if args.init_method == "xavier_uniform":
        kw_args['init_method'] = torch.nn.init.xavier_uniform_
        kw_args['scaled_init_method'] = torch.nn.init.xavier_uniform_
    
    # Return Transformer config.
    if getattr(args, "multi_latent_attention", False):
        return MLATransformerConfig(**kw_args)
    else:
        return TransformerConfig(**kw_args)

def load_yaml(yaml_path):
    print(f"warning using experimental yaml arguments feature, argparse arguments will be ignored")
    with open(yaml_path, "r") as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
        # Convert to nested namespace
        config_namespace = json.loads(json.dumps(config), object_hook=lambda item: SimpleNamespace(**item))
        # Add config location to namespace
        config_namespace.yaml_cfg = yaml_path
        return config_namespace

