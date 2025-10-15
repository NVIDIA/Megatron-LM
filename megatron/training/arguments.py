# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Megatron arguments."""

import argparse
import dataclasses
import json
import os
from pathlib import Path
import re
import types
import warnings

import torch
import torch.nn.functional as F
from packaging.version import Version as PkgVersion

from megatron.core.dist_checkpointing.validation import StrictHandling
from megatron.core.models.retro.utils import (
    get_config_path as get_retro_config_path,
    get_gpt_data_dir as get_retro_data_dir,
)
from megatron.core.rerun_state_machine import RerunStateMachine
from megatron.core.transformer import MLATransformerConfig, TransformerConfig
from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.heterogeneous.heterogeneous_config import (
    HeterogeneousTransformerConfig,
    MLPConfig,
)
from megatron.core.utils import (
    get_torch_version,
    is_te_min_version,
    is_torch_min_version,
)
from megatron.core.activations import squared_relu
from megatron.core.fusions.fused_bias_geglu import quick_gelu
from megatron.training.utils import (
    get_device_arch_version,
    update_use_dist_ckpt,
    print_rank_0,
    warn_rank_0,
)
from megatron.core.msc_utils import MultiStorageClientFeature

from megatron.core.quantization.utils import (
    kitchen_quantization_recipe_config,
    load_quantization_recipe,
)

def add_megatron_arguments(parser: argparse.ArgumentParser):
    """"Add Megatron-LM arguments to the given parser."""

    # Standard arguments.
    parser = _add_network_size_args(parser)
    parser = _add_regularization_args(parser)
    parser = _add_training_args(parser)
    parser = _add_rl_args(parser)
    parser = _add_initialization_args(parser)
    parser = _add_learning_rate_args(parser)
    parser = _add_checkpointing_args(parser)
    parser = _add_mixed_precision_args(parser)
    parser = _add_distributed_args(parser)
    parser = _add_validation_args(parser)
    parser = _add_data_args(parser)
    parser = _add_tokenizer_args(parser)
    parser = _add_autoresume_args(parser)
    parser = _add_biencoder_args(parser)
    parser = _add_vision_args(parser)
    parser = _add_moe_args(parser)
    parser = _add_mla_args(parser)
    parser = _add_heterogeneous_args(parser)
    parser = _add_logging_args(parser)
    parser = _add_straggler_detector_args(parser)
    parser = _add_workload_inspector_server_args(parser)
    parser = _add_inference_args(parser)
    parser = _add_transformer_engine_args(parser)
    parser = _add_retro_args(parser)
    parser = _add_experimental_args(parser)
    parser = _add_one_logger_args(parser)
    parser = _add_inprocess_restart_args(parser)
    parser = _add_ft_package_args(parser)
    parser = _add_config_logger_args(parser)
    parser = _add_rerun_machine_args(parser)
    parser = _add_msc_args(parser)
    parser = _add_kitchen_quantization_arguments(parser)
    parser = _add_sft_args(parser)

    return parser

def parse_args(extra_args_provider=None, ignore_unknown_args=False):
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description='Megatron-LM Arguments',
                                     allow_abbrev=False)

    parser = add_megatron_arguments(parser)

    # Custom arguments.
    if extra_args_provider is not None:
        parser = extra_args_provider(parser)

    # Parse.
    if ignore_unknown_args:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()

    # Experimental yaml
    if args.yaml_cfg is not None:
        from .yaml_arguments import load_yaml
        assert args.yaml_cfg and not args.use_legacy_models, \
            "Yaml config is not supported with legacy models."
        args = load_yaml(args.yaml_cfg)


    # Args from environment
    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))

    # Args to disable MSC
    if not args.enable_msc:
        MultiStorageClientFeature.disable()
        assert MultiStorageClientFeature.is_enabled() is False
        print('WARNING: The MSC feature is disabled.')

    return args


def validate_model_config_args_from_heterogeneous_config(args):
    """Validate model config arguments from heterogeneous config.

    This function takes model arguments and validates them based on a heterogeneous layer configuration.
    The heterogeneous config can be provided either as a path to a JSON file or as an encoded JSON string.

    The function enforces certain model architecture choices like SiLU activation, RMSNorm, grouped query attention,
    and RoPE positional embeddings. It also sets model dimensions like number of layers, hidden size, and attention heads
    based on the heterogeneous config.

    Args:
        args: Model configuration arguments to be overridden. Expected to have attributes:
            - heterogeneous_layers_config_path (str): Path to JSON config file
            - heterogeneous_layers_config_encoded_json (str): Encoded JSON config string

    Returns:
        None
    """
    if (
        args.heterogeneous_layers_config_path is None
        and args.heterogeneous_layers_config_encoded_json is None
    ):
        return

    if args.heterogeneous_layers_config_encoded_json is None:
        args.heterogeneous_layers_config_encoded_json = Path(
            args.heterogeneous_layers_config_path
        ).read_text()

    hf_config_dict = types.SimpleNamespace(**json.loads(args.heterogeneous_layers_config_encoded_json))

    assert hf_config_dict.hidden_act == "silu", (
        f"hidden_act in heterogeneous config is {hf_config_dict.hidden_act}, should be silu"
    )

    n_kv_heads_in_group = [
        config["attention"]["n_heads_in_group"] for config in hf_config_dict.block_configs 
        if config["attention"]["n_heads_in_group"] is not None
    ]
    assert all(num == n_kv_heads_in_group[0] for num in n_kv_heads_in_group), "num query head must be consistent across all layers"

    args_to_validate = {
        "swiglu": True,
        "normalization": "RMSNorm",
        "group_query_attention": True,
        "position_embedding_type": "rope",
        "rotary_percent": 1.0,
        "use_rope_scaling": True,
        "use_rotary_position_embeddings": True,
        "num_layers": hf_config_dict.num_hidden_layers,
        "hidden_size": hf_config_dict.hidden_size,
        "num_attention_heads": hf_config_dict.num_attention_heads,
        "untie_embeddings_and_output_weights": not hf_config_dict.tie_word_embeddings,
        "rotary_base": hf_config_dict.rope_theta,
        "rope_scaling_factor": hf_config_dict.rope_scaling["factor"],
        "num_query_groups": hf_config_dict.num_attention_heads // n_kv_heads_in_group[0],
    }

    incompatible_args = {}
    for key, value in args_to_validate.items():
        provided_value = getattr(args, key, None)
        if provided_value != value:
            incompatible_args[key] = (provided_value, value)

    if incompatible_args:
        incompatible_args_str = ', '.join([
            f"{k}: {provided_value} (provided) != {value} (expected)"
            for k, (provided_value, value) in incompatible_args.items()
        ])
        raise ValueError(
            f"Arguments differ from heterogeneous config: {incompatible_args_str}"
        )


def load_retro_config(retro_project_dir):
    '''Load Retro's config.json.'''

    # Retro config path.
    retro_config_path = get_retro_config_path(retro_project_dir)
    assert os.path.exists(retro_config_path), \
        "Retro project dir missing config.json."

    # Load retro config.
    with open(retro_config_path) as f:
        retro_config = types.SimpleNamespace(**json.load(f))

    return retro_config


def load_retro_args(args):
    """Load predefined args from Retro config (if applicable).

    When using Retro (or GPT for comparison purposes), data arguments are
    overridden by the saved config.json within the Retro project directory. This
    is to ensure that the data used for pretraining is consistent with the data
    that was preprocessed using the Retro preprocessing pipeline (see
    `tools/retro/preprocess_data.py`).
    """

    # Return if no project directory is specified.
    if args.retro_project_dir is None:
        return

    # Load retro config.
    retro_config = load_retro_config(args.retro_project_dir)

    # Retro data path is relative to project dir (via hard or soft links).
    data_dir = get_retro_data_dir(args.retro_project_dir)
    data_path = list(retro_config.retro_gpt_data_path)
    if len(data_path) % 2 == 0:
        for i in range(len(data_path) - 1, -1, -2):
            data_path[i] = os.path.join(data_dir, data_path[i])
    else:
        assert len(data_path) == 1
        data_path[0] = os.path.join(data_dir, data_path[0])

    # Update args.
    args.data_cache_path = retro_config.retro_gpt_data_cache_path
    args.data_path = data_path if args.data_path is None else args.data_path
    args.eval_interval = retro_config.retro_gpt_eval_interval
    args.eval_iters = retro_config.retro_gpt_eval_iters
    args.global_batch_size = retro_config.retro_gpt_global_batch_size
    args.max_position_embeddings = retro_config.retro_gpt_seq_length
    args.merge_file = os.path.join(
        args.retro_project_dir,
        retro_config.retro_gpt_merge_file,
    ) if retro_config.retro_gpt_merge_file is not None else None
    args.seed = retro_config.retro_gpt_seed
    args.seq_length = retro_config.retro_gpt_seq_length
    args.tokenizer_model = os.path.join(
        args.retro_project_dir,
        retro_config.retro_gpt_tokenizer_model,
    ) if retro_config.retro_gpt_tokenizer_model is not None else None
    args.tokenizer_type = retro_config.retro_gpt_tokenizer_type
    args.train_samples = retro_config.retro_gpt_train_samples
    args.vocab_file = os.path.join(
        args.retro_project_dir,
        retro_config.retro_gpt_vocab_file,
    ) if retro_config.retro_gpt_vocab_file is not None else None

    # Retro-specific args.
    args.retro_block_size = retro_config.retro_block_size
    args.retro_chunk_length = retro_config.retro_gpt_chunk_length
    args.retro_neighbor_dirs = retro_config.retro_neighbor_dirs
    args.retro_split_preprocessing = retro_config.retro_gpt_split
    args.retro_bert_tokenizer_type = retro_config.retro_bert_tokenizer_type
    args.retro_bert_vocab_file = retro_config.retro_bert_vocab_file

def _eval_pattern(pattern):
    """ Validate and evaluate a string containing a Python list expression """
    assert isinstance(pattern, str)

    # validate input, only allow comma, digits, [, ], (, ), +, and *
    if bool(re.compile(r'[^,\d\[\]\(\)\+\*]').search(pattern)):
        raise ValueError(f"Invalid pattern: {pattern}")

    return eval(pattern)

def no_rope_freq_type(x):
    """ Controls which layers to skip performing Rotary Position Embedding.
    - An integer N: Represents a 1:N ratio, meaning RoPE is skipped every N-1 layers.
    - A string "N": Same as above, but provided as a string
    - A string containing a Python list expression that defines a custom pattern, e.g.:
      "([0]*3+[1]*1)*3" evaluates to [0,0,0,1,0,0,0,1,0,0,0,1]
      where 1 indicates rope is skipped on the layer.
      This allows defining arbitrary patterns of rope skipping.
      The pattern length must match the total number of transformer layers.
      Examples:
          "([1]+[0]*23)": Only first layer has rope skipped for a 24-layer network.
          "([0]*3+[1]*1)*2": Every 4 layers the rope is skipped on the last layer. Repeat twice.
    """
    if x is None or isinstance(x, int):
        return x
    assert isinstance(x, str)
    if '[' in x:
        # it's a custom pattern
        return _eval_pattern(x)
    else:
        # it's a single int but in str
        return int(x)

def moe_freq_type(x):
    """Frequency between MoE layers and Dense layers.

    Accepts either:
    - An integer N: Represents a 1:N ratio, meaning one expert layer for every N-1 dense layers
    - A string "N": Same as above, but provided as a string
    - A string containing a Python list expression that defines a custom pattern, e.g.:
      "([1]*3+[0]*1)*3" evaluates to [1,1,1,0,1,1,1,0,1,1,1,0]
      where 1 indicates an expert layer and 0 indicates a dense layer.
      This allows defining arbitrary patterns of expert and dense layers.
      The pattern length must match the total number of transformer layers.
      Examples:
          "([0]+[1]*23)": 1 dense layer followed by 23 experts layers
          "([1]*3+[0]*2)*2": Three expert layers followed by two dense layers, repeated twice.
    """
    if isinstance(x, int):
        return x
    assert isinstance(x, str)
    if '[' in x:
        # it's a custom pattern
        return _eval_pattern(x)
    else:
        # it's a single int but in str
        return int(x)

def tuple_type(x):
    """
    Convert a string to a tuple of integers.
    Examples:
        "1,2,3" -> (1, 2, 3)
        "(1,2,3)" -> (1, 2, 3)
    """
    if x is None or isinstance(x, tuple):
        return x
    assert isinstance(x, str)
    return tuple(int(i) for i in x.strip('()').split(','))

def validate_args(args, defaults={}):

    # Temporary
    assert args.non_persistent_ckpt_type in ['global', 'local', None], \
        'Currently only global and local checkpoints are supported'
    if args.non_persistent_ckpt_type == 'local':
        try:
            from nvidia_resiliency_ext.checkpointing.local.ckpt_managers.local_manager import \
                LocalCheckpointManager
        except ModuleNotFoundError as e:
            raise RuntimeError('nvidia_resiliency_ext is required for local checkpointing') from e

    # validate model config args from heterogeneous config (if provided).
    validate_model_config_args_from_heterogeneous_config(args)

    # Load saved args from Retro (if applicable).
    load_retro_args(args)

    # Set args.use_dist_ckpt from args.ckpt_format.
    if args.use_legacy_models:
        assert args.ckpt_format == "torch", \
            "legacy model format only supports the 'torch' checkpoint format."
    update_use_dist_ckpt(args)

    total_model_size = args.tensor_model_parallel_size * args.pipeline_model_parallel_size * args.context_parallel_size

    # Total model size.
    assert args.world_size % total_model_size == 0, (
        f"world size ({args.world_size}) is not divisible by total_model_size ({total_model_size=})"
    )

    if args.attention_backend == AttnBackend.local:
        assert args.spec[0] == 'local' , '--attention-backend local is only supported with --spec local'

    # Pipeline model parallel size.
    args.transformer_pipeline_model_parallel_size = args.pipeline_model_parallel_size

    total_model_size = args.tensor_model_parallel_size * args.pipeline_model_parallel_size * args.context_parallel_size
    args.data_parallel_size = args.world_size // total_model_size

    # Batch size checks if running RL.
    if args.perform_rl_step:
        assert not (args.rl_remove_kv_cache_during_training and args.rl_offload_kv_cache_during_training), \
            "Cannot use both remove-kv-cache-during-training and offload-kv-cache-during-training"

        assert not (args.rl_partial_rollouts and args.rl_remove_kv_cache_during_training), \
            "Cannot use both partial-rollouts and remove-kv-cache-during-training"

        args.grpo_samples_per_iteration = args.grpo_prompts_per_step * args.grpo_group_size
        num_generated_samples_per_inference_iteration = (
            args.grpo_samples_per_iteration * args.grpo_iterations)

        # Ensure that the number of prompts we collect is a multiple of the global batch size.
        # TODO: Make this account for batch size rampup?
        assert num_generated_samples_per_inference_iteration % args.global_batch_size == 0, \
            f"grpo_group_size * grpo_prompts_per_step * grpo_iterations should be divisible by global_batch_size"

        # For now only exit/checkpoint on iterations where we generate data. We don't currently
        # have a way to checkpoint the generated data.
        num_training_iterations_per_inference_iteration = (
            num_generated_samples_per_inference_iteration // args.global_batch_size)
        if args.exit_interval is not None:
            assert args.exit_interval % num_training_iterations_per_inference_iteration == 0, \
                f"exit_interval should be divisible by number of global batches per inference iteration."
        if args.save_interval is not None:
            assert args.save_interval % num_training_iterations_per_inference_iteration == 0, \
                f"save_interval should be divisible by number of global batches per inference iteration."
        if args.rl_use_sequence_packing:
            assert args.seq_length <= args.rl_sequence_packing_bin_size, \
                f"rl_sequence_packing_bin_size should be larger than or equal to seq_length"

    if args.rank == 0:
        print('using world size: {}, data-parallel size: {}, '
              'context-parallel size: {}, '
              'hierarchical context-parallel sizes: {}, '
              'tensor-model-parallel size: {}, '
              'pipeline-model-parallel size: {}'.format(
                  args.world_size, args.data_parallel_size,
                  args.context_parallel_size,
                  args.hierarchical_context_parallel_sizes,
                  args.tensor_model_parallel_size,
                  args.pipeline_model_parallel_size), flush=True)

    # Checks.

    if args.hierarchical_context_parallel_sizes:
        from numpy import prod
        assert args.context_parallel_size == prod(args.hierarchical_context_parallel_sizes)
    if "a2a+p2p" in args.cp_comm_type:
        assert args.hierarchical_context_parallel_sizes is not None, \
        "--hierarchical-context-parallel-sizes must be set when a2a+p2p is used in cp comm"

    if args.expert_tensor_parallel_size is None:
        args.expert_tensor_parallel_size = args.tensor_model_parallel_size

    # Deprecated arguments.
    assert args.batch_size is None, '--batch-size argument is no longer ' \
        'valid, use --micro-batch-size instead'
    del args.batch_size
    assert args.warmup is None, '--warmup argument is no longer valid, use ' \
        '--lr-warmup-fraction instead'
    del args.warmup
    assert args.model_parallel_size is None, '--model-parallel-size is no ' \
        'longer valid, use --tensor-model-parallel-size instead'
    del args.model_parallel_size

    if args.checkpoint_activations:
        if args.rank == 0:
            print('--checkpoint-activations is no longer valid, use --recompute-activations, '
                  'or, for more control, --recompute-granularity and --recompute-method.')
        exit()
    del args.checkpoint_activations

    if args.recompute_activations:
        args.recompute_granularity = 'selective'
    del args.recompute_activations

    if args.enable_cuda_graph or args.external_cuda_graph:
        assert (
            args.cuda_graph_impl == "none"
        ), "Do not use --enable-cuda-graph or --external-cuda-graph with --cuda-graph-impl."
        assert (
            not args.enable_cuda_graph or not args.external_cuda_graph
        ), "--enable-cuda-graph and --external-cuda-graph cannot be enabled at the same time."

        if args.enable_cuda_graph:
            print_rank_0(
                '--enable-cuda-graph is deprecated, use --cuda-graph-impl=local instead.', args.rank
            )
            args.cuda_graph_impl = "local"
            del args.enable_cuda_graph
        if args.external_cuda_graph:
            print_rank_0(
                '--external-cuda-graph is deprecated, use --cuda-graph-impl=transformer_engine instead.',
                args.rank,
            )
            args.cuda_graph_impl = "transformer_engine"
            del args.external_cuda_graph

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

    if args.data_path is not None and args.split is None:
        legacy_default_split_value = '969, 30, 1'
        if args.rank == 0:
            print('WARNING: Please specify --split when using --data-path. Using legacy default value '
                  f'of "{legacy_default_split_value}"')
        args.split = legacy_default_split_value

    use_data_path = (args.data_path is not None) or (args.data_args_path is not None)
    if use_data_path:
        # Exactly one of the two has to be None if we use it.
        assert (args.data_path is None) or (args.data_args_path is None)
    use_per_split_data_path = any(
        elt is not None
        for elt in [args.train_data_path, args.valid_data_path, args.test_data_path]) or \
            args.per_split_data_args_path is not None
    if use_per_split_data_path:
         # Exactly one of the two has to be None if we use it.
        assert any(elt is not None
                   for elt in [args.train_data_path, args.valid_data_path, args.test_data_path]) is False or \
            args.per_split_data_args_path is None

    # Batch size.
    assert args.micro_batch_size is not None
    assert args.micro_batch_size > 0
    if args.global_batch_size is None:
        args.global_batch_size = args.micro_batch_size * args.data_parallel_size
        if args.rank == 0:
            print('setting global batch size to {}'.format(
                args.global_batch_size), flush=True)
    assert args.global_batch_size > 0

    # Uneven virtual pipeline parallelism
    assert (
        int(args.num_layers_per_virtual_pipeline_stage is not None)
        + int(args.num_virtual_stages_per_pipeline_rank is not None)
        + int(args.pipeline_model_parallel_layout is not None)
    ) <= 1, (
        'No more than one of the following arguments can be set at the same time: '
        '--num-layers-per-virtual-pipeline-stage, --num-virtual-stages-per-pipeline-rank,'
        '--pipeline-model-parallel-layout. '
        f'{args.num_layers_per_virtual_pipeline_stage=}, '
        f'{args.num_virtual_stages_per_pipeline_rank=}, '
        f'{args.pipeline_model_parallel_layout=}.'
    )

    if args.pipeline_model_parallel_layout is not None:
        # Parse the input flattened layout to a list and get the vpp size.
        # We will validate the layout more carefully in the TransformerConfig constructor.
        num_stages = PipelineParallelLayerLayout.get_num_stages_from_str(args.pipeline_model_parallel_layout)
        assert num_stages % args.pipeline_model_parallel_size == 0, (
            f"The length of pipeline_model_parallel_layout must be divisible"
            f" by pipeline_model_parallel_size ({num_stages=},"
            f" {args.pipeline_model_parallel_size=})"
        )
        args.virtual_pipeline_model_parallel_size = num_stages // args.pipeline_model_parallel_size
        if args.virtual_pipeline_model_parallel_size == 1:
            args.virtual_pipeline_model_parallel_size = None
    elif args.num_layers_per_virtual_pipeline_stage is not None or args.num_virtual_stages_per_pipeline_rank is not None:
        if args.num_virtual_stages_per_pipeline_rank is None:
            assert args.decoder_first_pipeline_num_layers is None and args.decoder_last_pipeline_num_layers is None, \
                'please use --num-virtual-stages-per-pipeline-rank to specify virtual pipeline parallel degree when enable uneven pipeline parallelism'
            if args.num_layers is not None:
                num_layers = args.num_layers
            else:
                num_layers = args.decoder_num_layers

            if args.account_for_embedding_in_pipeline_split:
                num_layers += 1

            if args.account_for_loss_in_pipeline_split:
                num_layers += 1

            assert num_layers % args.transformer_pipeline_model_parallel_size == 0, \
                'number of layers of the model must be divisible pipeline model parallel size'
            num_layers_per_pipeline_stage = num_layers // args.transformer_pipeline_model_parallel_size

            assert num_layers_per_pipeline_stage % args.num_layers_per_virtual_pipeline_stage == 0, \
                'number of layers per pipeline stage must be divisible number of layers per virtual pipeline stage'
            args.virtual_pipeline_model_parallel_size = num_layers_per_pipeline_stage // \
                args.num_layers_per_virtual_pipeline_stage
        else:
            args.virtual_pipeline_model_parallel_size = args.num_virtual_stages_per_pipeline_rank
        if args.virtual_pipeline_model_parallel_size == 1:
            args.virtual_pipeline_model_parallel_size = None
    else:
        args.virtual_pipeline_model_parallel_size = None

        if args.decoder_first_pipeline_num_layers is None and args.decoder_last_pipeline_num_layers is None:
            # Divisibility check not applicable for T5 models which specify encoder_num_layers
            # and decoder_num_layers.
            if args.num_layers is not None:
                num_layers = args.num_layers

                if args.account_for_embedding_in_pipeline_split:
                    num_layers += 1

                if args.account_for_loss_in_pipeline_split:
                    num_layers += 1

                assert num_layers % args.transformer_pipeline_model_parallel_size == 0, \
                    'Number of layers should be divisible by the pipeline-model-parallel size'
    
    if args.virtual_pipeline_model_parallel_size is not None:
        if args.overlap_p2p_comm:
            assert args.pipeline_model_parallel_size > 1, \
                'When interleaved schedule is used, pipeline-model-parallel size '\
                'should be greater than 1'
        else:
            assert args.pipeline_model_parallel_size > 2, \
                'When interleaved schedule is used and p2p communication overlap is disabled, '\
                'pipeline-model-parallel size should be greater than 2 to avoid having multiple '\
                'p2p sends and recvs between same 2 ranks per communication batch'
    else:
        # Overlap P2P communication is disabled if not using the interleaved schedule.
        args.overlap_p2p_comm = False
        args.align_param_gather = False
        # Only print warning if PP size > 1.
        if args.rank == 0 and args.pipeline_model_parallel_size > 1:
            print('WARNING: Setting args.overlap_p2p_comm and args.align_param_gather to False '
                'since non-interleaved schedule does not support overlapping p2p communication '
                'and aligned param AG')

    if args.rank == 0:
        print(f"Number of virtual stages per pipeline stage: {args.virtual_pipeline_model_parallel_size}")

    if args.overlap_param_gather:
        assert args.use_distributed_optimizer or args.use_megatron_fsdp, \
            '--overlap-param-gather only supported with distributed optimizer or megatron fsdp'
        assert args.overlap_grad_reduce, \
            'Must use --overlap-param-gather with --overlap-grad-reduce'
        assert not args.use_legacy_models, \
            '--overlap-param-gather only supported with MCore models'

    if args.use_torch_fsdp2:
        assert is_torch_min_version("2.4.0"), \
            'FSDP2 requires PyTorch >= 2.4.0 with FSDP 2 support.'
        assert args.pipeline_model_parallel_size == 1, \
            '--use-torch-fsdp2 is not supported with pipeline parallelism'
        assert args.expert_model_parallel_size == 1, \
            '--use-torch-fsdp2 is not supported with expert parallelism'
        assert not args.use_distributed_optimizer, \
            "--use-torch-fsdp2 is not supported with MCore's distributed optimizer"
        assert not args.gradient_accumulation_fusion, \
            '--use-torch-fsdp2 is not supported with gradient accumulation fusion'
        assert args.ckpt_format in ('torch_dist', 'torch_dcp'), \
            '--use-torch-fsdp2 requires --ckpt-format torch_dist or torch_dcp'
        assert args.untie_embeddings_and_output_weights, \
            '--use-torch-fsdp2 requires --untie-embeddings-and-output-weights'
        assert not args.fp16, \
            '--use-torch-fsdp2 not supported with fp16 yet'
        assert os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1", \
            'FSDP always requires CUDA_DEVICE_MAX_CONNECTIONS value large than one'

        if args.fp8_param_gather and is_te_min_version("2.0.0"):
            args.fp8_param_gather = False
            warn_rank_0(
                'FSDP2 FP8 param gather is not supported yet in TE 2.0, will fallback to bf16'
                'all_gather instead, turning off fp8_param_gather',
                args.rank,
            )
        if args.fp4_param and not is_te_min_version("2.7.0.dev0"):
            raise ValueError("--fp4-param requires Transformer Engine >= 2.7.0.dev0.")   

    if args.overlap_param_gather_with_optimizer_step:
        assert args.use_distributed_optimizer, \
            '--overlap-param-gather-with-optimizer-step only supported with distributed optimizer'
        assert args.overlap_param_gather, \
            'Must use --overlap-param-gather-with-optimizer-step with --overlap-param-gather'
        assert args.virtual_pipeline_model_parallel_size is not None, \
            '--overlap-param-gather-with-optimizer-step only supported with interleaved pipeline parallelism'
        assert not args.use_dist_ckpt, \
            '--overlap-param-gather-with-optimizer-step not supported with distributed checkpointing yet'

    dtype_map = {
        'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp8': torch.uint8,
    }
    map_dtype = lambda d: d if isinstance(d, torch.dtype) else dtype_map[d]

    args.main_grads_dtype = map_dtype(args.main_grads_dtype)
    args.main_params_dtype = map_dtype(args.main_params_dtype)
    args.exp_avg_dtype = map_dtype(args.exp_avg_dtype)
    args.exp_avg_sq_dtype = map_dtype(args.exp_avg_sq_dtype)

    if args.fp8_param_gather:
        assert args.use_distributed_optimizer or args.use_torch_fsdp2 or args.use_megatron_fsdp or not torch.is_grad_enabled(), \
            '--fp8-param-gather only supported with distributed optimizer, torch fsdp2, megatron fsdp, or inference mode'

    # FP4 and FP8 are mutually exclusive
    if args.fp4 and args.fp8:
        raise ValueError("--fp4-format and --fp8-format cannot be used simultaneously. Please choose one.")

    # FP4 param requires FP4 mode
    if args.fp4_param and not args.fp4:
        raise ValueError("--fp4-param-gather must be used together with --fp4-format.")
    
    # FP4 requires TE >= 2.7.0.dev0
    if args.fp4 and not is_te_min_version("2.7.0.dev0"):
        raise ValueError("--fp4-format requires Transformer Engine >= 2.7.0.dev0 for NVFP4BlockScaling support.")

    if args.use_megatron_fsdp:
        # NOTE: The flag `use_custom_fsdp` is deprecated and will be removed in future versions.
        #       Please use `use_megatron_fsdp` instead, as all functionality will be migrated there.
        #       Future updates will drop support for `use_custom_fsdp` to avoid confusion.
        args.use_custom_fsdp = True

        if args.data_parallel_sharding_strategy in ["optim_grads_params", "optim_grads"]:
            warn_rank_0(
                'Please make sure your TransformerEngine support FSDP + gradient accumulation fusion',
                args.rank,
            )

        if args.data_parallel_sharding_strategy == "optim_grads_params":
            assert args.check_weight_hash_across_dp_replicas_interval is None, \
                'check_weight_hash_across_dp_replicas_interval is not supported with optim_grads_params'

        assert os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1", \
            'FSDP always requires CUDA_DEVICE_MAX_CONNECTIONS value large than one'

        assert args.ckpt_format == "fsdp_dtensor", \
            "Megatron FSDP only supports fsdp_dtensor checkpoint format"

    # Parameters dtype.
    args.params_dtype = torch.float
    if args.fp16:
        assert not args.bf16
        args.params_dtype = torch.half
        # Turn off checking for NaNs in loss and grads if using dynamic loss scaling,
        # where NaNs in grads / loss are signal to the loss scaler.
        if not args.loss_scale:
            args.check_for_nan_in_loss_and_grad = False
            if args.rank == 0:
                print('WARNING: Setting args.check_for_nan_in_loss_and_grad to False since '
                      'dynamic loss scaling is being used')
    if args.bf16:
        assert not args.fp16
        args.params_dtype = torch.bfloat16
        # bfloat16 requires gradient accumulation and all-reduce to
        # be done in fp32.
        if args.accumulate_allreduce_grads_in_fp32:
            assert args.main_grads_dtype == torch.float32, \
                "--main-grads-dtype can only be fp32 when --accumulate-allreduce-grads-in-fp32 is set"

        if args.grad_reduce_in_bf16:
            args.accumulate_allreduce_grads_in_fp32 = False
        elif not args.accumulate_allreduce_grads_in_fp32 and args.main_grads_dtype == torch.float32:
            args.accumulate_allreduce_grads_in_fp32 = True
            if args.rank == 0:
                print('accumulate and all-reduce gradients in fp32 for '
                      'bfloat16 data type.', flush=True)
    if args.cuda_graph_impl == "local" and "full_iteration" in args.cuda_graph_scope:
        if not args.inference_dynamic_batching:
            assert not args.check_for_nan_in_loss_and_grad, \
            "--no-check-for-nan-in-loss-and-grad should be set with full_iteration CUDA graph"
        else:
            assert args.fp8 is None, \
            "fp8 is not supported with inference dynamic batching and full_iteration CUDA graph"

    if args.rank == 0:
        print('using {} for parameters ...'.format(args.params_dtype),
              flush=True)

    if args.dataloader_type is None:
        args.dataloader_type = 'single'

    # data
    assert args.num_dataset_builder_threads > 0

    # Consumed tokens.
    args.consumed_train_samples = 0
    args.skipped_train_samples = 0
    args.consumed_valid_samples = 0
    if args.rl_use_sequence_packing:
        args.consumed_train_bins = 0

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
        assert args.encoder_num_layers is not None, \
            'either num-layers or encoder-num-layers should be specified'
        args.num_layers = args.encoder_num_layers

    # Check required arguments.
    required_args = ['num_layers', 'hidden_size', 'num_attention_heads',
                     'max_position_embeddings']
    for req_arg in required_args:
        _check_arg_is_not_none(args, req_arg)

    # Checks.
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

    if args.seq_length is not None and args.context_parallel_size > 1:
        assert args.seq_length % (args.context_parallel_size * 2) == 0, \
            'seq-length should be a multiple of 2 * context-parallel-size ' \
            'if context-parallel-size > 1.'

    if args.seq_length is not None:
        assert args.encoder_seq_length is None
        args.encoder_seq_length = args.seq_length
    else:
        assert args.encoder_seq_length is not None
        args.seq_length = args.encoder_seq_length

    if args.seq_length is not None:
        assert args.max_position_embeddings >= args.seq_length, \
            f"max_position_embeddings ({args.max_position_embeddings}) must be greater than " \
            f"or equal to seq_length ({args.seq_length})."
    if args.decoder_seq_length is not None:
        assert args.max_position_embeddings >= args.decoder_seq_length
    if args.lr is not None:
        assert args.min_lr <= args.lr
    if args.save is not None:
        assert args.save_interval is not None
        assert args.save_interval > 0
        if args.save_retain_interval is not None:
            assert args.save_retain_interval > 0
            assert args.save_retain_interval % args.save_interval == 0
    # Mixed precision checks.
    if args.fp16_lm_cross_entropy:
        assert args.fp16, 'lm cross entropy in fp16 only support in fp16 mode.'
    if args.fp32_residual_connection:
        assert args.fp16 or args.bf16, \
            'residual connection in fp32 only supported when using fp16 or bf16.'

    if args.moe_grouped_gemm:
        assert args.bf16, 'Currently GroupedGEMM for MoE only supports bf16 dtype.'
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
        args.no_persist_layer_norm = True
        if args.rank == 0:
            print('Persistent fused layer norm kernel is supported from '
                  'pytorch v1.11 (nvidia pytorch container paired with v1.11). '
                  'Defaulting to no_persist_layer_norm=True')

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
        assert is_torch_min_version("1.10.0a0"), \
            'distributed recompute activations are supported for pytorch ' \
            'v1.10 and above (Nvidia Pytorch container >= 21.07). Current ' \
            f'pytorch version is v{get_torch_version()}.'

    if args.recompute_granularity == 'selective':
        assert args.recompute_method is None, \
            'recompute method is not yet supported for ' \
            'selective recomputing granularity'

    # disable sequence parallelism when tp=1
    # to avoid change in numerics when
    # sequence_parallelism is enabled.
    if args.tensor_model_parallel_size == 1:
        if args.sequence_parallel:
            warn_rank_0(
                "Disabling sequence parallelism because tensor model parallelism is disabled",
                args.rank,
            )
        args.sequence_parallel = False

    if args.tp_comm_overlap:
        assert args.sequence_parallel == True, 'Tensor parallel communication/GEMM overlap can happen only when sequence parallelism is enabled'

    # disable async_tensor_model_parallel_allreduce when
    # model parallel memory optimization is enabled
    if (args.tensor_model_parallel_size > 1 or args.context_parallel_size > 1) \
        and get_device_arch_version() < 10:
        # CUDA_DEVICE_MAX_CONNECTIONS requirement no longer exists since the Blackwell architecture
        if args.use_torch_fsdp2 or args.use_megatron_fsdp:
            fsdp_impl = "Torch-FSDP2" if args.use_torch_fsdp2 else "Megatron-FSDP"
            warn_rank_0(
                f"Using tensor model parallelism or context parallelism with {fsdp_impl} together. "
                "Try not to using them together since they require different CUDA_MAX_CONNECTIONS "
                "settings for best performance. sequence parallelism requires setting the "
                f"environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 while {fsdp_impl} "
                "requires not setting CUDA_DEVICE_MAX_CONNECTIONS=1 for better parallelization.",
                args.rank,
            )
        elif args.overlap_moe_expert_parallel_comm:
            warn_rank_0(
                "For Hopper and before, try not to use tensor model parallelism or context parallelism with overlap_moe_expert_parallel_comm. "
                "Using tensor/context model parallelism requires setting the environment "
                "variable CUDA_DEVICE_MAX_CONNECTIONS to 1 to maximize the performance. "
                "While overlap_moe_expert_parallel_comm requires setting a larger CUDA_DEVICE_MAX_CONNECTIONS "
                "for better parallelization. If you want to use both, you can set CUDA_DEVICE_MAX_CONNECTIONS to 1 or 32, "
                "which depends on which parallelization you want to prioritize.",
                args.rank,
            )
        else:
            assert os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') == "1", \
                "Using tensor model parallelism or context parallelism require setting the environment variable " \
                "CUDA_DEVICE_MAX_CONNECTIONS to 1"

    # Setting FSDP communication groups for high priority streams for Blackwell and later architectures
    # Assigning high priority to communication streams ensures that communication kernels are scheduled
    # with higher priority, minimizing the exposed communication when it is overlapped with other computation kernels.
    if args.use_torch_fsdp2 or args.use_megatron_fsdp and get_device_arch_version() >= 10:
        if 'dp_cp' not in args.high_priority_stream_groups:
            args.high_priority_stream_groups.append('dp_cp')
        if args.expert_model_parallel_size  > 1 and 'ep_dp' not in args.high_priority_stream_groups:
            args.high_priority_stream_groups.append('ep_dp')

    # Disable bias gelu fusion if we are disabling bias altogether
    if not args.add_bias_linear:
        args.bias_gelu_fusion = False

    # Keep the 'add bias' args in sync; add_qkv_bias is more targeted.
    if args.add_bias_linear:
        args.add_qkv_bias = True

    # Retro checks.
    if args.retro_add_retriever:

        # Train samples should be auto-loaded.
        assert args.train_samples is not None, \
            "args.train_samples should be auto-loaded from the retro config."

        # Sequence parallelism unsupported.
        assert not args.sequence_parallel, \
            "retro currently does not support sequence parallelism."

        # Pipeline parallelism unsupported.
        assert args.pipeline_model_parallel_size == 1, \
            "retro currently does not support pipeline parallelism."

    if args.decoupled_lr is not None or args.decoupled_min_lr is not None:
        assert not args.use_legacy_models, \
            '--decoupled-lr and --decoupled-min-lr is not supported in legacy models.'

    # Legacy RoPE arguments
    if args.use_rotary_position_embeddings:
        args.position_embedding_type = 'rope'
    if args.rotary_interleaved and args.use_legacy_models:
        raise RuntimeError('--rotary-interleaved is not supported in legacy models.')
    if args.position_embedding_type != 'rope':
        args.apply_rope_fusion = False

    # Would just need to add 'NoPE' as a position_embedding_type to support this, but for now
    # don't allow it to keep things simple
    if not args.add_position_embedding and args.position_embedding_type != 'rope':
        raise RuntimeError('--no-position-embedding is deprecated, use --position-embedding-type')

    # Relative position embeddings arguments
    if args.position_embedding_type == 'relative':
        assert (
            args.transformer_impl == "transformer_engine"
        ), 'Local transformer implementation currently does not support attention bias-based position embeddings.'

    # MultiModal rotary embeddings arguments
    if args.position_embedding_type == "mrope":
        assert args.mrope_section is not None, \
            '--mrope-section should be set when using --position-embedding-type mrope.'

    # MoE Spec check
    if args.num_experts == 0:
        args.num_experts = None
    if args.num_experts is not None:
        assert args.spec is None, "Model Spec must be None when using MoEs"
    if args.num_experts is not None and args.moe_ffn_hidden_size is None:
        args.moe_ffn_hidden_size = args.ffn_hidden_size
        print("Warning: moe_ffn_hidden_size is not set, using ffn_hidden_size for MoE instead.")

    # Context parallel
    if args.context_parallel_size > 1:
        assert not args.use_legacy_models, "Context parallelism is not supported in legacy models."

    # Expert parallelism check
    if args.expert_model_parallel_size  > 1:
        assert args.num_experts is not None, "num_experts must be non None to use expert model parallelism"
        assert args.num_experts % args.expert_model_parallel_size == 0, \
            "Number of experts should be a multiple of expert model parallel_size."
        assert not args.fp16, \
            "Expert parallelism is not supported with fp16 training."

    # MoE router check
    if isinstance(args.moe_router_load_balancing_type, list) and len(args.moe_router_load_balancing_type) == 1:
        args.moe_router_load_balancing_type = args.moe_router_load_balancing_type[0]
    if isinstance(args.moe_aux_loss_coeff, list) and len(args.moe_aux_loss_coeff) == 1:
        args.moe_aux_loss_coeff = args.moe_aux_loss_coeff[0]

    # Distributed checkpointing checks
    if args.use_dist_ckpt and args.use_legacy_models:
        raise RuntimeError('--use-dist-ckpt is not supported in legacy models.')

    # torch_dcp (torch.distributed.checkpoint) checkpointing format checks.
    if args.ckpt_format == "torch_dcp":
        assert args.use_torch_fsdp2, "--ckpt-format torch_dcp is only tested with FSDP."
        assert args.tensor_model_parallel_size <= 1, \
            "--ckpt-format torch_dcp is not tested with megatron tensor parallelism."
        assert args.pipeline_model_parallel_size <= 1, \
            "--ckpt-format torch_dcp is not tested with megatron pipeline parallelism."

    # fsdp_dtensor checkpointing format checks.
    if args.ckpt_format == "fsdp_dtensor":
        assert args.use_megatron_fsdp, "--ckpt-format fsdp_dtensor is only tested with Megatron FSDP."

    # Data blend checks
    assert args.mock_data + \
           bool(args.data_path) + \
           any([args.train_data_path, args.valid_data_path, args.test_data_path]) \
           <= 1, "A single data source must be provided in training mode, else None"

    # Deterministic mode
    if args.deterministic_mode:
        assert not args.use_flash_attn, "Flash attention can not be used in deterministic mode."
        assert not args.cross_entropy_loss_fusion, "Cross Entropy Fusion is currently not deterministic."

        all_reduce_choices = ["Tree", "Ring", "CollnetDirect", "CollnetChain", "^NVLS"]
        assert os.getenv("NCCL_ALGO", -1) != -1 and os.getenv("NCCL_ALGO") in all_reduce_choices, \
            f"NCCL_ALGO must be one of {all_reduce_choices}."

        torch.use_deterministic_algorithms(True)

    # Update the printed args to reflect that `apply_query_key_layer_scaling` also controls `attention_softmax_in_fp32`
    if args.apply_query_key_layer_scaling:
        args.attention_softmax_in_fp32 = True

    if args.result_rejected_tracker_filename is not None:
        # Append to passed-in args.iterations_to_skip.
        iterations_to_skip_from_file = RerunStateMachine.get_skipped_iterations_from_tracker_file(
            args.result_rejected_tracker_filename
        )
        args.iterations_to_skip.extend(iterations_to_skip_from_file)

    # Make sure all functionality that requires Gloo process groups is disabled.
    if not args.enable_gloo_process_groups:
        if args.use_distributed_optimizer:
            # If using distributed optimizer, must use distributed checkpointing.
            # Legacy checkpointing uses Gloo process groups to collect full distributed
            # optimizer state in the CPU memory of DP rank 0.
            assert args.use_dist_ckpt

            if args.dist_ckpt_optim_fully_reshardable:
                assert not args.distrib_optim_fully_reshardable_mem_efficient, \
                    '--distrib-optim-fully-reshardable-mem-efficient requires -enable-gloo-process-groups'


    # Checkpointing
    if args.ckpt_fully_parallel_save_deprecated and args.rank == 0:
        print('--ckpt-fully-parallel-save flag is deprecated and has no effect.'
              ' Use --no-ckpt-fully-parallel-save to disable parallel save.')
    if (
        args.use_dist_ckpt
        and not args.ckpt_fully_parallel_save
        and args.use_distributed_optimizer
        and args.rank == 0
    ):
        print('Warning: With non-parallel ckpt save and DistributedOptimizer,'
              ' it will be impossible to resume training with different parallelism.'
              ' Consider removing flag --no-ckpt-fully-parallel-save.')
    if args.use_dist_ckpt_deprecated and args.rank == 0:
        print('--use-dist-ckpt is deprecated and has no effect.'
              ' Use --ckpt-format to select the checkpoint format.')
    if args.dist_ckpt_format_deprecated and args.rank == 0:
        print('--dist-ckpt-format is deprecated and has no effect.'
              ' Use --ckpt-format to select the checkpoint format.')

    if args.load_main_params_from_ckpt:
        assert args.no_load_optim, '--load-main-params-from-ckpt must be used with --no-load-optim.'

    # Inference args
    if args.inference_batch_times_seqlen_threshold > -1:
        assert args.pipeline_model_parallel_size > 1, \
            "--inference-batch-times-seqlen-threshold requires setting --pipeline-model-parallel-size > 1."
        assert (
            args.cuda_graph_impl == "none"
        ), "Pipeline-parallel microbatched inference is incompatible with CUDA graphs"

    if args.inference_dynamic_batching:
        assert args.inference_dynamic_batching_buffer_size_gb is not None
        assert args.inference_dynamic_batching_block_size % 256 == 0, "block size should be a multiple of 256"
        assert args.inference_dynamic_batching_buffer_guaranteed_fraction is not None

    # MoE upcycling check
    if args.moe_use_upcycling:
        assert args.save is not None, "When using upcycling, the --save option must be specified."
        if not args.no_load_optim:
            args.no_load_optim = True
            print('Warning: disabling --no-load-optim for upcycling.')
        if not args.no_load_rng:
            args.no_load_rng = True
            print('Warning: disabling --no-load-rng for upcycling.')

    # Optimizer CPU offload check
    if args.optimizer_cpu_offload:
        assert args.use_precision_aware_optimizer, (
            "The optimizer cpu offload must be used in conjunction with `--use-precision-aware-optimizer`, "
            "as the hybrid device optimizer reuses the code path of this flag."
        )
        assert not args.fp8_param_gather or args.fp8_recipe == "delayed", (
            "When `--fp8-param-gather` is enabled, the optimizer cpu offload "
            "must be used in conjunction with `--fp8-recipe delayed`."
        )

    if args.non_persistent_ckpt_type == "local":
        assert args.non_persistent_local_ckpt_dir is not None, "Tried to use local checkpointing without specifying --local-ckpt-dir!"
    if args.replication:
        assert args.replication_jump is not None, "--replication requires the value of --replication-jump!"
        assert args.non_persistent_ckpt_type == "local", f"--replication requires args.non_persistent_ckpt_type == 'local', but got: {args.non_persistent_ckpt_type}"
    elif args.replication_jump:
        print("Warning: --replication-jump was specified despite not using replication. Ignoring.")
        args.replication_jump = None

    if args.delay_wgrad_compute:
        assert args.transformer_impl == 'transformer_engine', \
            "Delaying wgrad compute is only supported with transformer_engine implementation"
        if args.overlap_grad_reduce:
            assert is_te_min_version("2.8.0"), (
                "overlap_grad_reduce is only supported with TE >= 2.8.0 when enabling delay_wgrad_compute"
            )
        if not args.gradient_accumulation_fusion:
            assert is_te_min_version("2.7.0"), (
                "disabling gradient_accumulation_fusion is only supported with TE >= 2.7.0 "
                "when enabling delay_wgrad_compute"
            )

    if args.mtp_num_layers:
        assert not args.use_legacy_models, "The legacy Megatron models does not support Multi-Token Prediction (MTP)."
        assert args.position_embedding_type == "rope" or args.position_embedding_type == "none", (
            f"Multi-Token Prediction (MTP) is not supported with {args.position_embedding_type} position embedding type."
            + f"The supported position embedding types are rope and none."
        )

    if args.cpu_offloading_num_layers > 0:
        args.cpu_offloading = True

    # CUDA Graphs
    if args.cuda_graph_impl != "none":
        if args.transformer_impl == 'transformer_engine' and not args.te_rng_tracker:
            args.te_rng_tracker = True
            warn_rank_0("te_rng_tracker is not enabled, enabling it for CUDA graphs.", args.rank)
        assert "expandable_segments:True" not in os.getenv("PYTORCH_CUDA_ALLOC_CONF", ""), (
            "expandable_segments:True may not be safe when using CUDA Graphs with some specific parallel settings. "
            "The training may crash with illegal memory access."
        )
        assert (
            args.recompute_granularity != 'full'
        ), 'recompute_granularity must not be full when CUDA Graphs are enabled.'

        if args.cuda_graph_impl == "local":
            if args.cuda_graph_scope:
                assert args.cuda_graph_scope == [
                    "full_iteration"
                ], "For local cuda graph implementation, the only valid value for --cuda-graph-scope is full_iteration. To use other scopes, use --cuda-graph-impl=transformer_engine."

        if args.cuda_graph_impl == "transformer_engine":
            assert (
                "full_iteration" not in args.cuda_graph_scope
            ), "To use full iteration cuda graph, please use --enable-cuda-graph instead of --external-cuda-graph."

            if not args.cuda_graph_scope:
                # Set the scope default to the whole layer. This will work for a dense model but may raise error for MoE.
                # So the user should explicitly set the scope for MoE models.
                if args.hybrid_override_pattern is not None:
                    if '*' in args.hybrid_override_pattern:
                        args.cuda_graph_scope.append('attn')
                    if 'M' in args.hybrid_override_pattern:
                        args.cuda_graph_scope.append('mamba')
                    if 'E' in args.hybrid_override_pattern:
                        args.cuda_graph_scope.append('moe')
                    if '-' in args.hybrid_override_pattern:
                        args.cuda_graph_scope.append('mlp')
                elif args.num_experts is None or args.num_experts <= 1:
                    args.cuda_graph_scope = ['attn', 'mlp']
                elif args.moe_layer_freq == 1 or (
                    isinstance(args.moe_layer_freq, list) and 0 not in args.moe_layer_freq
                ):
                    args.cuda_graph_scope = ['attn', 'moe']
                else:
                    args.cuda_graph_scope = ['attn', 'mlp', 'moe']

            for scope in args.cuda_graph_scope:
                assert scope in [
                    'attn',
                    'mlp',
                    'moe',
                    'moe_router',
                    'moe_preprocess',
                    'mamba',
                ], f"--cuda-graph-scope should be attn, mlp, moe, moe_router, moe_preprocess, or mamba, got {args.cuda_graph_scope}."
            assert (
                'moe' not in args.cuda_graph_scope or 'moe_router' not in args.cuda_graph_scope
            ), 'cuda_graph_scope must not contain both moe and moe_router.'
            if 'moe_preprocess' in args.cuda_graph_scope:
                assert (
                    'moe_router' in args.cuda_graph_scope
                ), 'moe_preprocess cuda graph is only supported with moe_router cuda graph.'
            if args.num_experts is None or args.num_experts <= 1:
                assert (
                    'moe' not in args.cuda_graph_scope and 'moe_router' not in args.cuda_graph_scope
                ), 'moe cuda graph is only supported for MoE.'
            else:
                if (
                    args.moe_layer_freq == 1
                    or (isinstance(args.moe_layer_freq, list) and 0 not in args.moe_layer_freq)
                ) and (
                    args.hybrid_override_pattern is None or '-' not in args.hybrid_override_pattern
                ):
                    assert (
                        'mlp' not in args.cuda_graph_scope
                    ), 'mlp cuda graph is only supported for dense layers, but not found in the model.'
                if args.moe_token_dispatcher_type in ['flex', 'allgather']:
                    assert (
                        'moe' not in args.cuda_graph_scope
                    ), 'moe cuda graph is not supported for flex or allgather token dispatcher.'
                elif args.moe_token_dispatcher_type == 'alltoall' and (
                    args.moe_expert_capacity_factor is None
                    or not args.moe_pad_expert_input_to_capacity
                ):
                    assert (
                        'moe' not in args.cuda_graph_scope
                    ), 'moe cuda graph is only supported with drop-padding MoE.'
                    if (
                        args.moe_expert_capacity_factor is not None
                        or args.moe_router_padding_for_fp8
                    ):
                        assert (
                            'moe_preprocess' not in args.cuda_graph_scope
                        ), 'moe_preprocess cuda graph is not supported when there are DtoH copies and synchronizations in the preprocess step.'

    # Print arguments.
    _print_args("arguments", args)

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


def core_transformer_config_from_args(args, config_class=None):

    # Config class.
    config_class = config_class or TransformerConfig

    if args.multi_latent_attention:
        config_class = MLATransformerConfig
    
    if args.heterogeneous_layers_config_path is not None:
        assert not args.multi_latent_attention, "Multi latent attention with heterogeneous layers is not supported."
        config_class = HeterogeneousTransformerConfig

    # Translate args to core transformer configuration
    kw_args = {}
    for f in dataclasses.fields(config_class):
        if hasattr(args, f.name):
            kw_args[f.name] = getattr(args, f.name)
    kw_args['persist_layer_norm'] = not args.no_persist_layer_norm
    kw_args['layernorm_zero_centered_gamma'] = args.apply_layernorm_1p
    kw_args['layernorm_epsilon'] = args.norm_epsilon
    kw_args['deallocate_pipeline_outputs'] = True
    kw_args['pipeline_dtype'] = args.params_dtype
    kw_args['batch_p2p_comm'] = not args.overlap_p2p_comm
    kw_args['num_moe_experts'] = args.num_experts
    kw_args['rotary_interleaved'] = args.rotary_interleaved
    kw_args['num_layers_in_first_pipeline_stage']= args.decoder_first_pipeline_num_layers
    kw_args['num_layers_in_last_pipeline_stage']= args.decoder_last_pipeline_num_layers
    kw_args['fp8_param'] = args.fp8_param_gather
    if args.swiglu:
        kw_args['activation_func'] = F.silu
        kw_args['gated_linear_unit'] = True
        kw_args['bias_activation_fusion'] = args.bias_swiglu_fusion
    else:
        kw_args['bias_activation_fusion'] = args.bias_gelu_fusion
    if args.squared_relu:
        assert not args.swiglu
        kw_args['activation_func'] = squared_relu
    elif args.quick_geglu:
        assert not args.swiglu
        kw_args['gated_linear_unit'] = True
        kw_args['activation_func'] = quick_gelu
    if args.init_method_xavier_uniform:
        kw_args['init_method'] = torch.nn.init.xavier_uniform_
        kw_args['scaled_init_method'] = torch.nn.init.xavier_uniform_
    if args.group_query_attention:
        kw_args['num_query_groups'] = args.num_query_groups
    else:
        kw_args['num_query_groups'] = None
    kw_args['config_logger_dir'] = args.config_logger_dir
    if args.rope_type is None:
        # Pop 'rope_type' to let the config class use the default value.
        kw_args.pop('rope_type', None)
    else:
        assert (args.multi_latent_attention or args.rope_type == 'rope'), (
            f'Common attention only support rope_type="rope", but got {args.rope_type}.'
        )

    if len(args.cp_comm_type) == 1:
        kw_args['cp_comm_type'] = args.cp_comm_type[0]
    if args.is_hybrid_model:
        kw_args['is_hybrid_model'] = args.is_hybrid_model

    kw_args['inference_sampling_seed'] = args.seed

    # handle quantization config
    # NOTE: Kitchen arguments are only added to the namespace when
    # Kitchen library is available.
    if hasattr(args, "kitchen_config_file") and args.kitchen_config_file is not None:
        kw_args['use_kitchen'] = True
        kw_args['quant_recipe'] = load_quantization_recipe(args.kitchen_config_file)
    elif hasattr(args, 'kitchen_recipe_number') and args.kitchen_recipe_number is not None:
        kw_args['use_kitchen'] = True
        kw_args['quant_recipe'] = kitchen_quantization_recipe_config(args.kitchen_recipe_number)


    # Return config.
    return config_class(**kw_args)


def _add_transformer_engine_args(parser):
    group = parser.add_argument_group(title='Transformer-Engine')

    group.add_argument('--fp8-format', default=None,
                       choices=['e4m3', 'hybrid'],
                       help='Which fp8 format scheme to use for FP8 tensors in the forward and backward pass',
                       dest='fp8')
    # per tensor current scaling recipe selection
    group.add_argument('--fp8-recipe', default='delayed',
                       choices=['tensorwise', 'delayed', 'mxfp8', 'blockwise'],
                       help='Which fp8 recipe to use for FP8 tensors in the forward and backward pass',
                       dest='fp8_recipe')
    # delayed scaling only configs
    group.add_argument('--fp8-margin', type=int, default=0,
                       help='Scaling margin for fp8',
                       dest='fp8_margin')
    group.add_argument('--fp8-interval', type=int, default=1,
                       help='DEPRECATED. This flag is ignored. Scaling update interval for fp8',
                       dest='fp8_interval')
    group.add_argument('--fp8-amax-history-len', type=int, default=1,
                       help='Number of steps for which amax history is recorded per tensor',
                       dest='fp8_amax_history_len')
    group.add_argument('--fp8-amax-compute-algo', default='most_recent',
                       choices=['most_recent', 'max'],
                       help='Algorithm for computing amax from history',
                       dest='fp8_amax_compute_algo')
    group.add_argument('--no-fp8-wgrad', action='store_false',
                       help='Execute wgrad in higher precision even for FP8 runs',
                       dest='fp8_wgrad')
    group.add_argument('--transformer-impl', default='transformer_engine',
                       choices=['local', 'transformer_engine'],
                       help='Which Transformer implementation to use.')
    group.add_argument('--fp8-param-gather', action='store_true',
                       help='Keep the compute param in fp8 (do not use any other intermediate '
                            'dtype) and perform the param all-gather in fp8.')
    group.add_argument('--first-last-layers-bf16', action='store_true',
                       help='Construct first and last layers in bf16 when doing FP8 training.')
    group.add_argument('--num-layers-at-start-in-bf16', type=int, default=1,
                       help='Number of layers at start to construct in bf16 when --first-last-layers-bf16 is enabled.')
    group.add_argument('--num-layers-at-end-in-bf16', type=int, default=1,
                       help='Number of layers at end to construct in bf16 when --first-last-layers-bf16 is enabled.')
    
    # FP4 related arguments
    group.add_argument('--fp4-format', default=None,
                       choices=['e2m1'],
                       help='Which nvfp4 format scheme to use for FP4 tensors in the forward and backward pass',
                       dest='fp4')
    group.add_argument('--fp4-recipe', default='nvfp4',
                       choices=['nvfp4'],
                       help='Which fp4 recipe to use for FP4 tensors in the forward and backward pass',
                       dest='fp4_recipe')
    group.add_argument('--fp4-param-gather', action='store_true',
                       help='Keep the compute param in fp4 (do not use any other intermediate '
                            'dtype) and perform the param all-gather in fp4.',
                       dest='fp4_param')
    group.add_argument('--te-rng-tracker', action='store_true', default=False,
                       help='Use the Transformer Engine version of the random number generator. '
                            'Required for CUDA graphs support.')
    group.add_argument('--inference-rng-tracker', action='store_true', default=False,
                       help='Use a random number generator configured for inference.')
    return parser

def _add_inference_args(parser):
    group = parser.add_argument_group(title='inference')

    group.add_argument('--inference-batch-times-seqlen-threshold',
                       type=int, default=-1,
                       help='If (batch-size * sequence-length) is smaller than this threshold'
                       'then batches will not be split up for pipelining.'
                       'Requires setting --pipeline-model-parallel-size > 1.'
                       'Setting this to -1 indicates that batch pipelining is not used.')
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
    group.add_argument('--flash-decode', default=False, action="store_true",
                       help='Whether to use the flash decoding kernel.')
    group.add_argument('--enable-cuda-graph', default=False, action="store_true",
                       help='Deprecated. Use --cuda-graph-impl=local instead. '
                       'Use local implementation of CUDA graph capture and replay. '
                       '--cuda-graph-scope=\"full_iteration\" enables whole iteration CUDA graph. ')
    group.add_argument("--cuda-graph-warmup-steps", type=int, default=3,
                       help="Number of CUDA graph warmup steps")
    group.add_argument('--external-cuda-graph', action='store_true',
                       help='Deprecated. Use --cuda-graph-impl=transformer_engine instead. '
                       'Use TE make_graphed_callables() to capture the CUDA graph. '
                       'Use --cuda-graph-scope=\"attn\", \"mlp\", \"moe\", \"moe_router\", \"moe_preprocess\", \"mamba\" for partial capture. ')
    group.add_argument('--cuda-graph-impl', type=str, default='none',
                       choices=['none', 'local', 'transformer_engine'],
                       help='Determines the CUDA graph capture implementation. '
                       '"none": no CUDA graph. '
                       '"local": capture the CUDA graph using MCore local implementation. --cuda-graph-scope=\"full_iteration\" enables whole iteration CUDA graph. '
                       '"transformer_engine": capture the CUDA graph using TE make_graphed_callables().')
    group.add_argument('--cuda-graph-scope', nargs='+', type=str, default=[],
                       help='Determines the CUDA graphs capturing scope. '
                       'choices: "attn", "mlp", "moe", "moe_router", "moe_preprocess", "mamba", "full_iteration". '
                       '"attn": captures operations in TransformerLayer._forward_attention(). '
                       '"mlp": captures operations in TransformerLayer._forward_mlp() for a dense layer. '
                       '"moe": captures operations in TransformerLayer._forward_mlp() for a MoE layer. '
                       '"moe_router": captures operations in TransformerLayer._forward_mlp() up to MoELayer.router(), '
                       'including the shared experts if they are not overlapped with EP comm. '
                       '"moe_preprocess": captures operations in MoELayer.preprocess(). Must be used together with "moe_router". '
                       '"mamba": captures the mamba layer. '
                       '"full_iteration": captures a whole iteration. '
                       'full_iteration scope is only supported with --cuda-graph-impl=local, other scopes are only supported with --cuda-graph-impl=transformer_engine. '
                       'If not specified, the default scope is to capture the whole Transformer layer.')
    group.add_argument('--use-legacy-static-engine', action='store_true', default=False,
                       help='Use legacy static engine. (Current static engine uses dynamic engine under the hood)',
                       dest='use_legacy_static_engine')
    group.add_argument('--inference-max-requests', type=int, default=8,
                       help='Maximum number of requests for inference.',
                       dest='inference_max_batch_size')
    group.add_argument('--inference-max-seq-length', type=int, default=2560,
                       help='Maximum sequence length expected for inference (prefill + decode).',
                       dest='inference_max_seq_length')
    group.add_argument('--inference-max-batch-size', type=int, default=None,
                       help='Maximum batch size for inference.',
                       dest='inference_max_batch_size')
    group.add_argument('--inference-dynamic-batching',
                       action='store_true', default=False,
                       help='Enable dynamic batching mode.')
    group.add_argument('--inference-dynamic-batching-buffer-size-gb',
                       type=float, default=40.,
                       help='Total buffer size (GB) allocated for the block-level KV '
                       'memory.')
    group.add_argument('--inference-dynamic-batching-block-size',
                       type=int, default=256,
                       help='KV cache block size. '
                       'It should be a multiple of 256')
    group.add_argument('--inference-dynamic-batching-buffer-guaranteed-fraction',
                       type=float, default=0.2,
                       help='Space is reserved within the inference context '
                       'memory buffer to guarantee that a minimum number of '
                       'active requests will always be able to run to '
                       'completion. This is to avoid the context being deadlocked '
                       'by paused requests.')
    group.add_argument('--inference-dynamic-batching-buffer-overflow-factor',
                       type=float, default=None,
                       help='Scaling factor over the memory buffer size for auto '
                       'computing `max_requests` and `max_tokens`. This scaling '
                       'factor is used for fitting more requests and tokens in '
                       'the memory buffer than it can safely hold, which in turn '
                       'increases throughput.')
    group.add_argument('--inference-dynamic-batching-max-requests-override',
                       type=int, default=None,
                       help='If set, this overrides the max requests as computed '
                       'from `--inference-dynamic-batching-buffer-overflow-factor`.')
    group.add_argument('--inference-dynamic-batching-max-tokens-override',
                       type=int, default=None,
                       help='If set, this overrides the max tokens as computed '
                       'from `--inference-dynamic-batching-buffer-overflow-factor`.')
    group.add_argument('--inference-dynamic-batching-num-cuda-graphs',
                       type=int, default=16,
                       help='Maximum number of cuda graphs to capture, where the '
                       'cuda graph batch sizes range from 1 to `max_requests`. '
                       '(See `dynamic_context.py` for details on how '
                       '`max_requests` is computed). Due to rounding, the actual '
                       'number of cuda graphs may not equal this argument.')
    group.add_argument('--inference-dynamic-batching-track-paused-request-events',
                       action='store_true',
                       help='Track paused request ids by adding \'paused\' events '
                       'to each request\'s event history. This has a very minor '
                       'impact on latency.')
    group.add_argument('--decode-only-cuda-graphs',
                       action='store_true', default=False,
                       help='Only use cuda graphs for decode-only steps, not prefill and mixed steps.')
    group.add_argument('--inference-dynamic-batching-unified-memory-level',
                       type=int, default=0, choices=[0, 1],
                       help='Set unified memory usage within the dynamic '
                       'inference context. The levels are: 0) no unified memory, '
                       '1) allocate `memory_buffer` in unified memory. '
                       'Eventually, additional levels will be included to '
                       'control other tensors within the context.')
    group.add_argument('--symmetric-ar-type', type=str, default=None,
                       choices=['two_shot', "one_shot", "multimem_all_reduce", None],
                       help='What type of symmetric all reduce to use. The default is none which is no use of symetric memory')
    group.add_argument('--nccl-all-reduce-for-prefill',
                       action='store_true', default=False,
                       help='When using symmeric all reduce kernels this will use regular nccl kernels for prefill. This can be more effecient when prefill is large as the nccl kernels can be more bandwith optimized')
    group.add_argument('--mlp-chunks-for-prefill', type=int, default=1,
                       help='Number of chunks along sequence dimension for MLP '
                       'computation during prefill')
    group.add_argument('--disable-chunked-prefill', default=False, action="store_true",
                       help='Disable chunked prefill (chunked prefill is enabled by default).')  
    return parser


def _add_retro_args(parser):
    group = parser.add_argument_group(title='retro')

    group.add_argument('--retro-project-dir', default=None,
                       help='Retro project directory, which contains the '
                       'preprocessed data for pretraining. This directory '
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
    group.add_argument("--retro-attention-gate", type=float, default=1,
                       help="Gated cross attention.")
    group.add_argument("--retro-no-verify-neighbor-count", action="store_false",
                       dest="retro_verify_neighbor_count",
                       help="Skip verifying that len(GPT dataset) == len(saved "
                       "neighbors).")

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
    group.add_argument('--hidden-size', type=int, default=None,
                       help='Transformer hidden size.')
    group.add_argument('--ffn-hidden-size', type=int, default=None,
                       help='Transformer Feed-Forward Network hidden size. '
                       'This is set to 4*hidden-size if not provided')
    group.add_argument('--num-attention-heads', type=int, default=None,
                       help='Number of transformer attention heads.')
    group.add_argument('--attention-backend', type=lambda attn_backend: AttnBackend[attn_backend], default=AttnBackend.auto, choices = list(AttnBackend), help='Attention backend to use (flash,fused,unfused,local,auto). Defaults to auto')
    group.add_argument('--kv-channels', type=int, default=None,
                       help='Projection weights dimension in multi-head '
                       'attention. This is set to '
                       '   args.hidden_size // args.num_attention_heads '
                       'if not provided.')
    group.add_argument('--group-query-attention', action='store_true',
                          help='Use group-query attention.')
    group.add_argument('--num-query-groups', type=int, default=1)
    group.add_argument('--softmax-type', type=str, default='vanilla',
                       choices=['learnable', 'vanilla', 'off-by-one'],
                       help='Type of softmax to use for the attention. Supports both a fixed offset and '
                       'learnable offset.')
    group.add_argument('--window-size', type=tuple_type, default=None,
                       help='Window size for window attention. If not provided, '
                            'window attention will be disabled.')
    group.add_argument('--window-attn-skip-freq', type=moe_freq_type, default=None,
                       help='Frequency of layers to skip window attention. Accepts either: '
                            '- An integer N: Represents a (N-1):1 ratio, meaning one full attention layer '
                            'after (N-1) SWA layers. '
                            '- A string containing a Python list expression that defines a custom pattern, '
                            'e.g.: "[1,1,1,0]*3" evaluates to [1,1,1,0,1,1,1,0,1,1,1,0] '
                            'where 1 indicates SWA and 0 indicates full attention. ')
    group.add_argument('--max-position-embeddings', type=int, default=None,
                       help='Maximum number of position embeddings to use. '
                       'This is the size of position embedding.')
    group.add_argument('--position-embedding-type', type=str, default='learned_absolute',
                        choices=['learned_absolute', 'rope', 'mrope', 'relative', 'none'],
                        help='Position embedding type.')
    group.add_argument('--relative-attention-num-buckets', type=int, default=32,
                        help='Number of buckets for relative position embeddings.')
    group.add_argument('--relative-attention-max-distance', type=int, default=128,
                        help='Maximum distance for relative position embeddings calculation.')
    group.add_argument('--use-rotary-position-embeddings', action='store_true',
                       help='Use rotary positional embeddings or not. '
                       'Deprecated: use --position-embedding-type')
    group.add_argument('--rotary-base', type=int, default=10000,
                       help='Base to use for rotary positional embeddings, default 10000')
    group.add_argument('--rotary-percent', type=float, default=1.0,
                       help='Percent of rotary dimension to use, default 100%%')
    group.add_argument('--rotary-interleaved', action='store_true',
                          help='Use interleaved rotary embedding.')
    group.add_argument('--rotary-seq-len-interpolation-factor', type=int, default=None,
                       help='Sequence length interpolation factor for rotary embeddings.')
    group.add_argument('--use-rope-scaling', action='store_true',
                       help='Apply rope scaling as used in llama3.x')
    group.add_argument('--rope-scaling-factor', type=float, default=8.0,
                       help='Rope scaling factor in llama3.x models')
    group.add_argument('--no-rope-freq', type=no_rope_freq_type, default=None,
                       help='Controls which layers to skip performing Rotary Position Embedding. Accepts either: '
                            '- An integer N: Represents a 1:N ratio, meaning RoPE is skipped every N-1 layers. '
                            '- A string containing a Python list expression that defines a custom pattern, e.g.: '
                            '"([0]*3+[1]*1)*3" evaluates to [0,0,0,1,0,0,0,1,0,0,0,1] '
                            'where 1 indicates no-rope layer. This patten is equivalent to --no-rope-freq=4.'
                            'By default this is disabled and set to None, indicating RoPE will be performed'
                            'on every layer.'
                       )
    group.add_argument('--no-position-embedding',
                       action='store_false',
                       help='Disable position embedding. Deprecated: use --position-embedding-type',
                       dest='add_position_embedding')
    group.add_argument('--mrope-section', nargs='+', type=int, default=None,
                       help='Multimodal rope section is for channel dimension, empty by default.')
    group.add_argument('--make-vocab-size-divisible-by', type=int, default=128,
                       help='Pad the vocab size to be divisible by this value.'
                       'This is added for computational efficieny reasons.')
    group.add_argument('--normalization', default='LayerNorm',
                       choices=['LayerNorm', 'RMSNorm'],
                       help='Which normalization technique to use.')
    group.add_argument('--norm-epsilon', type=float, default=1e-5,
                       help='Epsilon for layer norm and RMS norm.')
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
    group.add_argument('--quick-geglu', action='store_true',
                       help='Use quick geglu activation instead of default gelu')
    group.add_argument('--activation-func-clamp-value', type=float, default=None,
                       help='Clamp the output of the linear_fc1 in the activation function. Only used when '
                            'activation_func is quick_gelu.')
    group.add_argument('--glu-linear-offset', type=float, default=0.0,
                       help='Offset term in the GLU activation function: activation_func(x[0]) * (x[1] + offset). '
                            'Only used when gated_linear_unit is True')
    group.add_argument('--onnx-safe', type=bool, required=False,
                       help='Use workarounds for known problems with '
                       'Torch ONNX exporter')
    group.add_argument('--bert-no-binary-head', action='store_false',
                       help='Disable BERT binary head.',
                       dest='bert_binary_head')
    group.add_argument('--untie-embeddings-and-output-weights', action='store_true',
                       help='Untie embeddings and output weights.')
    group.add_argument('--multi-latent-attention', action='store_true',
                       help='Use multi-latent attention for model.')
    group.add_argument('--mtp-num-layers', type=int, default=None,
                       help='Number of Multi-Token Prediction (MTP) Layers.'
                       'MTP extends the prediction scope to multiple future tokens at each position.'
                       'This MTP implementation sequentially predict additional tokens '
                       'by using D sequential modules to predict D additional tokens.')
    group.add_argument('--mtp-loss-scaling-factor', type=float, default=0.1,
                       help='Scaling factor of Multi-Token Prediction (MTP) loss. '
                       'We compute the average of the MTP losses across all depths, '
                       'and multiply it the scaling factor to obtain the overall MTP loss, '
                       'which serves as an additional training objective.')
    return parser


def _add_straggler_detector_args(parser):
    group = parser.add_argument_group(title='straggler')
    group.add_argument('--log-straggler', action='store_true',
                       help='If set, tracks and logs straggler per GPU.')
    group.add_argument('--disable-straggler-on-startup', action='store_true',
                       help='If set, StragglerDetector is disabled on startup.')
    group.add_argument('--straggler-ctrlr-port', type=int, default=65535,
                       help='Port number to toggle StragglerDetector on/off at runtime')
    group.add_argument('--straggler-minmax-count', type=int, default=1,
                       help='Number of ranks to report with high/low estimated throughput')
    return parser

def _add_workload_inspector_server_args(parser):
    group = parser.add_argument_group(title='workload inspector')
    group.add_argument('--run-workload-inspector-server', action='store_true',
                       help='If set, enables workload inspector server for on-demand profiling.')
    return parser

def _add_inprocess_restart_args(parser):
    group = parser.add_argument_group(title='In-process restart')

    group.add_argument('--inprocess-restart', action='store_true',
                       help='Enables in-process restart.')

    group.add_argument('--inprocess-max-iterations', default=None, type=int,
                       help='Maximum number of in-process restart iterations.')
    group.add_argument('--inprocess-monitor-thread-interval', default=1.0, type=float,
                       help='Monitoring interval (in seconds) for the monitoring thread.')
    group.add_argument('--inprocess-monitor-process-interval', default=1.0, type=float,
                       help='Monitoring interval (in seconds) for the monitoring process.')
    group.add_argument('--inprocess-progress-watchdog-interval', default=1.0, type=float,
                       help='Interval (in seconds) for automatic progress watchdog timestamp '
                       'updates.')
    group.add_argument('--inprocess-heartbeat-interval', default=30, type=float,
                       help='Monitoring interval (in seconds) for detecting unresponsive ranks.')

    group.add_argument('--inprocess-soft-timeout', default=60, type=float,
                       help='Soft progress timeout (in seconds).')
    group.add_argument('--inprocess-hard-timeout', default=90, type=float,
                       help='Hard progress timeout (in seconds).')
    group.add_argument('--inprocess-heartbeat-timeout', default=60, type=float,
                       help='Timeout (in seconds) for a missing rank detection heartbeat.')

    group.add_argument('--inprocess-barrier-timeout', default=120, type=float,
                       help='Timeout (in seconds) for internal distributed barrier')
    group.add_argument('--inprocess-completion-timeout', default=120, type=float,
                       help='Timeout (in seconds) for barrier on completion on all ranks')

    group.add_argument('--inprocess-last-call-wait', default=1, type=float,
                       help='Time interval (in seconds) for other ranks to report concurrent '
                       'terminal failures.')
    group.add_argument('--inprocess-termination-grace-time', default=1, type=float,
                       help='Interval (in seconds) between SIGTERM and SIGKILL issued on hard '
                       'timeout')

    group.add_argument('--inprocess-granularity', default='node', type=str,
                       choices=['node', 'rank'],
                       help='Granularity for in-process restart.')
    group.add_argument('--inprocess-active-world-size',
                       default=int(os.getenv('WORLD_SIZE', '1')), type=int,
                       help='The number of ranks initially executing the workload. '
                       'The remaining ranks from the allocation are set aside '
                       'as warm reserve.')
    group.add_argument('--inprocess-empty-cuda-cache', action='store_true',
                       help='Release all unoccupied cached GPU memory on every in-process restart.')
    return parser

def _add_one_logger_args(parser):
    group = parser.add_argument_group(title='one logger')
    group.add_argument('--no-one-logger', action='store_false',
                       help='If set, disable using one_logger to track E2E metrics'
                       'Note that one_logger is an internal tool and not '
                       'available externally. For installation, please go to '
                       'https://confluence.nvidia.com/display/MLWFO/Package+Repositories'
                       'for more details',
                       dest='enable_one_logger')
    group.add_argument('--one-logger-project', type=str, default='megatron-lm',
                       help='The one-logger project name. Will ignore if '
                       '--no-one-logger is set')
    group.add_argument('--one-logger-run-name', type=str, default=None,
                       help='The one-logger run name displayed. Will ignore if '
                       '--no-one-logger is set')
    group.add_argument('--one-logger-async', action='store_true',
                       help='If set, forces one_logger to use async mode.')
    group.add_argument('--app-tag-run-name', type=str, default=None,
                       help='Jobs belonging to same training run, suppose to '
                       'have the same name. It will be used to track progress of '
                       'a training done over multiple different jobs')
    group.add_argument('--app-tag-run-version', type=str, default='0.0.0',
                       help='The version of the training of which current job is '
                       'part of. It will be used to track the changes in the '
                       'application side which might change the performance '
                       'baseline')
    return parser


def _add_ft_package_args(parser):
    group = parser.add_argument_group(title='ft_package')
    group.add_argument('--enable-ft-package', action='store_true',
                       help='If set, Fault Tolerance package is enabled. '
                       'Note: This feature is for Nvidia internal use only.')
    group.add_argument('--calc-ft-timeouts', action='store_true',
                       help='If set, FT package will try to automatically compute the timeouts. '
                       'Note: This feature is for Nvidia internal use only.')
    return parser


def _add_config_logger_args(parser):
    group = parser.add_argument_group(title='config logger')
    group.add_argument('--config-logger-dir', type=str, default='',
                       help='If set, will dump all configs to --config-logger-dir',
                       dest='config_logger_dir')
    return parser


def _add_logging_args(parser):
    group = parser.add_argument_group(title='logging')

    group.add_argument('--log-params-norm', action='store_true',
                       help='If set, calculate and log parameters norm.')
    group.add_argument('--log-num-zeros-in-grad', action='store_true',
                       help='If set, calculate and log the number of zeros in gradient.')
    group.add_argument('--log-throughput', action='store_true',
                       help='If set, calculate and log throughput per GPU.')
    group.add_argument('--log-progress', action='store_true',
                       help='If set, log progress (in terms of number of processed tokens and '
                       'number of floating-point operations) to progress.txt file in checkpoint '
                       'directory.')
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
    group.add_argument('--log-energy', action='store_true',
                       help='If set, log energy consumption (in Joules)')
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
                       'and summaries before one of the "add" calls forces a '
                       'flush to disk.')
    group.add_argument('--log-timers-to-tensorboard', action='store_true',
                       help='If set, write timers to tensorboard.')
    group.add_argument('--no-log-loss-scale-to-tensorboard',
                       action='store_false',
                       help='Disable loss-scale logging to tensorboard.',
                       dest='log_loss_scale_to_tensorboard')
    group.add_argument('--log-validation-ppl-to-tensorboard',
                       action='store_true',
                       help='If set, write validation perplexity to '
                       'tensorboard.')
    group.add_argument('--log-memory-to-tensorboard',
                       action='store_true',
                       help='Enable memory logging to tensorboard.')
    group.add_argument('--log-world-size-to-tensorboard',
                       action='store_true',
                       help='Enable world size logging to tensorboard.')
    group.add_argument('--wandb-project', type=str, default='',
                       help='The wandb project name. Ignore wandb by default.')
    group.add_argument('--wandb-entity', type=str, default='',
                       help='The wandb entity name. It is useful when '
                       'there are multiple sub-projects in a project. '
                       'https://community.wandb.ai/t/how-do-i-decide-which-account-private-or-team-to-upload-the-run-to/5704 '
                       'Ignore wandb by default.')    
    group.add_argument('--wandb-exp-name', type=str, default='',
                       help='The wandb experiment name.')
    group.add_argument('--wandb-save-dir', type=str, default='',
                       help='Path to save the wandb results locally.')
    group.add_argument('--logging-level', type=int, default=None,
                       help='Set default logging level')
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


def _add_rl_args(parser):
    group = parser.add_argument_group(title='rl')
    group.add_argument('--perform-rl-step', action='store_true',
                       help="Use the RL training step.")
    group.add_argument('--rl-prompts-per-eval', type=int, default=32,
                       help='Number of prompts to evaluate for for each RL task.'
                        'This evaluation can be very expensive when using environments' 
                        'that evaluate pass@k so we default to a lower number.')
    # TODO(rkirby): allow for "complete" evaluation when --rl-prompts-per-eval is set to -1
    group.add_argument('--grpo-prompts-per-step', type=int, default=32,
                       help="Number of GRPO groups (G in the paper).")
    group.add_argument('--grpo-group-size', type=int, default=2,
                       help="Number of samples per a GRPO group.")
    group.add_argument('--grpo-iterations', type=int, default=2,
                       help="Number of iterations per a GRPO implementation.")
    # As in DAPO, we keep upper/lower eps different.
    # To have a vanilla GRPO, set them to be the same.
    group.add_argument('--grpo-clamp-eps-lower', type=float, default=0.01,
                       help="Lower GRPO clipping bound.")
    group.add_argument('--grpo-clamp-eps-upper', type=float, default=0.01,
                       help="Upper GRPO clipping bound. In vanilla implementation, equals to the lower one.")
    group.add_argument('--grpo-kl-beta', type=float, default=0.001,
                       help="KL term weight in the GRPO loss.")
    group.add_argument('--grpo-entropy-term-weight', type=float, default=0.0,
                       help="Entropy term weight in GRPO loss.")
    group.add_argument('--grpo-filter-groups-with-same-reward', action='store_true',
                       help="Filter groups with same reward.")
    group.add_argument('--grpo-default-temperature', type=float, default=1.0,
                       help="Default temperature for model inference.")
    group.add_argument('--grpo-default-top-p', type=float, default=0,
                       help="Default top-p for model inference.")
    group.add_argument('--langrl-inference-server-type', type=str,
                       choices=['inplace_megatron', 'inplace_megatron_chat'], default='inplace_megatron',
                       help="Type of inference server to use.")
    group.add_argument('--langrl-inference-server-conversation-template', type=str, default=None,
                       help="Conversation template, if using a chat server.")
    group.add_argument('--langrl-external-server', action=argparse.BooleanOptionalAction, required=False, default=False)
    group.add_argument('--langrl-env-config', type=str, default=None,
                       help="Path to YAML config file for RL environment configuration.")
    group.add_argument('--rl-offload-optimizer-during-inference', action='store_true',
                       help='Offload optimizer state to CPU during inference/rollout to save GPU memory')
    group.add_argument('--rl-offload-kv-cache-during-training', action=argparse.BooleanOptionalAction, default=False,
                       help='Offload KV cache to CPU during training to save GPU memory')
    group.add_argument('--rl-remove-kv-cache-during-training', action=argparse.BooleanOptionalAction, default=False,
                       help='Remove KV cache during training to save GPU memory')
    group.add_argument('--rl-reset-cuda-graphs', action=argparse.BooleanOptionalAction, type=bool, default=False,
                       help='Reset CUDA graphs between inference/training to save GPU memory')
    group.add_argument('--rl-partial-rollouts', action=argparse.BooleanOptionalAction, default=False,
                       help='If set, use partial rollouts.')
    group.add_argument('--rl-inference-logprobs-is-correction', action=argparse.BooleanOptionalAction, type=bool, default=False,
                       help='If set, use inference logprobs in importance sampling correction of the loss.')
    group.add_argument('--rl-importance-sampling-truncation-coef', type=float, default=None,
                       help="If --inference-logprobs-is-correction is on and this coefficient is set, apply truncation for the IS correction at GRPO loss.")
    group.add_argument('--rl-calculate-intra-group-similarity', action=argparse.BooleanOptionalAction, default=False,
                       help='If set, calculate the intra-group similarity of rollouts.')
    group.add_argument('--rl-use-sequence-packing', action='store_true',
                       help='Enable sequence packing')
    group.add_argument('--rl-sequence-packing-bin-size', type=int, default=8192,
                       help='Override bin size for sequence packing.')
    group.add_argument('--rl-sequence-packing-algo', type=str, default='fifo',
                       choices=['fifo', 'round-robin'],
                       help='Algorithm for distributing packed bins across ranks. '
                            'fifo: first-in-first-out sequential distribution, '
                            'round-robin: distribute bins cyclically across ranks for better load balancing')
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
                       '   --rampup-batch-size 16 8 300000 \\ '
                       '   --global-batch-size 1024'
                       'will start with global batch size 16 and over '
                       ' (1024 - 16) / 8 = 126 intervals will increase'
                       'the batch size linearly to 1024. In each interval'
                       'we will use approximately 300000 / 126 = 2380 samples.')
    group.add_argument('--decrease-batch-size-if-needed', action='store_true', default=False,
                       help='If set, decrease batch size if microbatch_size * dp_size'
                       'does not divide batch_size. Useful for KSO (Keep Soldiering On)'
                       'to continue making progress if number of healthy GPUs (and'
                       'corresponding dp_size) does not support current batch_size.'
                       'Old batch_size will be restored if training is re-started with'
                       'dp_size that divides batch_size // microbatch_size.')
    group.add_argument('--recompute-activations', action='store_true',
                       help='recompute activation to allow for training '
                       'with larger models, sequences, and batch sizes.')
    group.add_argument('--recompute-granularity', type=str, default=None,
                       choices=['full', 'selective'],
                       help='Checkpoint activations to allow for training '
                       'with larger models, sequences, and batch sizes. '
                       'It is supported at two granularities 1) full: '
                       'whole transformer layer is recomputed, '
                       '2) selective: submodules set in --recompute-modules '
                       'are recomputed, default is core_attn.')
    group.add_argument('--no-check-for-nan-in-loss-and-grad', action='store_false',
                       help='Check for NaNs in loss and grad',
                       dest='check_for_nan_in_loss_and_grad')
    group.add_argument('--check-for-spiky-loss', action='store_true',
                       help='Check for spiky loss',
                       dest='check_for_spiky_loss')
    group.add_argument('--check-for-large-grads', action='store_true',
                       help='Check for unexpectedly large grads',
                       dest='check_for_large_grads')
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
    group.add_argument('--recompute-num-layers', type=int, default=None,
                       help='1) uniform: the number of Transformer layers in each '
                       'uniformly divided recompute unit, '
                       '2) block: the number of individual Transformer layers '
                       'to recompute within each pipeline stage.')
    group.add_argument('--recompute-modules', nargs='*', type=str, default=None,
                       help='The submodules to recompute. '
                       'choices: "core_attn", "moe_act", "layernorm", "mla_up_proj", '
                       '         "mlp", "moe", "shared_experts". '
                       'default: ["core_attn"].'
                       '"core_attn": recompute the core attention part of the transformer layer. '
                       '"moe_act": recompute the MoE MLP activation function. '
                       '"layernorm": recompute the input_layernorm and pre_mlp_layernorm. '
                       '"mla_up_proj": recompute the MLA up projection and RoPE applying parts.'
                       '"mlp": recompute the dense MLP layer.'
                       '"moe": recompute the MoE layer.'
                       '"shared_experts": recompute the shared experts in the MoE layer.'
                       '"moe_act", "layernorm", and "mla_up_proj" use output-discarding checkpointing, '
                       '"core_attn", "mlp", "moe", and "shared_experts" use normal checkpointing.')
    group.add_argument('--cpu-offloading-num-layers', type=int, default=0,
                       help='The number of Transformer layers to offload to CPU.')
    group.add_argument('--no-clone-scatter-output-in-embedding', action='store_false',
                       help='If not set, clone the output of the scatter in embedding layer to GC original tensor.',
                       dest='clone_scatter_output_in_embedding')
    group.add_argument('--profile', action='store_true',
                       help='Enable nsys profiling. When using this option, nsys '
                       'options should be specified in commandline. An example '
                       'nsys commandline is `nsys profile -s none -t nvtx,cuda '
                       '-o <path/to/output_file> --force-overwrite true '
                       '--capture-range=cudaProfilerApi '
                       '--capture-range-end=stop`.')
    group.add_argument('--profile-step-start', type=int, default=10,
                       help='Global step to start profiling.')
    group.add_argument('--profile-step-end', type=int, default=12,
                       help='Global step to stop profiling.')
    group.add_argument('--iterations-to-skip', nargs='+', type=int, default=[],
                       help='List of iterations to skip, empty by default.')
    group.add_argument('--result-rejected-tracker-filename', type=str, default=None,
                       help='Optional name of file tracking `result_rejected` events.')
    group.add_argument('--disable-gloo-process-groups', action='store_false',
                       dest='enable_gloo_process_groups',
                       help='Disables creation and usage of Gloo process groups.')
    group.add_argument('--use-pytorch-profiler', action='store_true',
                       help='Use the built-in pytorch profiler. '
                       'Useful if you wish to view profiles in tensorboard.',
                       dest='use_pytorch_profiler')
    group.add_argument('--profile-ranks', nargs='+', type=int, default=[0],
                       help='Global ranks to profile.')
    group.add_argument('--record-memory-history', action="store_true", default=False,
                       help='Record memory history in last rank.')
    group.add_argument('--memory-snapshot-path', type=str, default="snapshot.pickle",
                       help='Specifies where to dump the memory history pickle.')
    group.add_argument('--tp-comm-overlap', action='store_true', help='Enables the '
                       ' overlap of Tensor parallel communication and GEMM kernels.')
    group.add_argument('--tp-comm-overlap-cfg', type=str, default=None,
                       help='Config file when tp_comm_overlap is enabled.')
    group.add_argument('--disable-tp-comm-overlap-ag', action='store_false',
                       help=('Disables the All-Gather overlap with GEMM by '
                             'pipelining the GEMM and All-Gather.'),
                       dest='tp_comm_overlap_ag')
    group.add_argument('--disable-tp-comm-overlap-rs', action='store_false',
                       help=('Disables the Reduce-Scatter overlap with GEMM by '
                             'pipelining the GEMM and Reduce-Scatter.'),
                       dest='tp_comm_overlap_rs')
    group.add_argument('--tp-comm-overlap-rs-dgrad', action='store_true',
                       help = 'Enables the Reduce-Scatter overlap with dgrad GEMM.',
                       dest='tp_comm_overlap_rs_dgrad')
    group.add_argument('--disable-tp-comm-bulk-dgrad', action='store_false',
                       help='Disables the All-Gather overlap with bprop activation gradient GEMM.',
                       dest='tp_comm_bulk_dgrad')
    group.add_argument('--disable-tp-comm-bulk-wgrad', action='store_false',
                       help='Disables the Reduce-Scatter overlap with bprop weight gradient GEMM.',
                       dest='tp_comm_bulk_wgrad')
    group.add_argument('--tp-comm-bootstrap-backend', default='nccl', type=str,
                       choices=['nccl', 'mpi', 'gloo'],
                       help='Set the bootstrapping backend of Tensor parallel communications.')
    group.add_argument('--use-cpu-initialization', action='store_true',
                       default=None,
                       help='If set, initialize weights on the CPU. This eliminates init differences based on tensor parallelism.')
    group.add_argument('--empty-unused-memory-level', default=0, type=int,
                       choices=[0, 1, 2],
                       help='Call torch.cuda.empty_cache() each iteration '
                       '(training and eval), to reduce fragmentation.'
                       '0=off, 1=moderate, 2=aggressive.')
    group.add_argument('--deterministic-mode', action='store_true',
                       help='Choose code that has deterministic execution. This usually '
                       'means slower execution, but is good for debugging and testing.')
    group.add_argument('--check-weight-hash-across-dp-replicas-interval', type=int, default=None,
                       help='Interval to check weight hashes are same across DP replicas. If not specified, weight hashes not checked.')
    group.add_argument('--calculate-per-token-loss', action='store_true',
                       help=('Scale cross entropy loss by the number of non-padded tokens in the '
                             'global batch, versus the default behavior of assuming all tokens are non-padded.'))
    group.add_argument('--train-sync-interval', type=int, default=None,
                       help='Training CPU-GPU synchronization interval, to ensure that CPU is not running too far ahead of GPU.')

    # deprecated
    group.add_argument('--checkpoint-activations', action='store_true',
                       help='Checkpoint activation to allow for training '
                       'with larger models, sequences, and batch sizes.')
    group.add_argument('--train-iters', type=int, default=None,
                       help='Total number of iterations to train over all '
                       'training runs. Note that either train-iters or '
                       'train-samples should be provided.')
    group.add_argument('--train-samples', type=int, default=None,
                       help='Total number of samples to train over all '
                       'training runs. Note that either train-iters or '
                       'train-samples should be provided.')
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
    group.add_argument('--no-bias-swiglu-fusion', action='store_false',
                       help='Disable bias and swiglu fusion, the fusion is '
                       'available only when using megatron-core.',
                       dest='bias_swiglu_fusion')
    group.add_argument('--use-fused-weighted-squared-relu', action='store_true',
                       help='Use fused weighted squared relu when using MoE.')
    group.add_argument('--no-bias-dropout-fusion', action='store_false',
                       help='Disable bias and dropout fusion.',
                       dest='bias_dropout_fusion')
    group.add_argument('--no-rope-fusion', action='store_false',
                       help='Disable rope fusion, the fusion is available '
                       'only when using megatron-core.',
                       dest='apply_rope_fusion')
    group.add_argument('--rope-type', type=str, default=None,
                      choices=['rope', 'yarn'],
                      help='Type of rope to use. Note that MLA takes yarn by default, '
                      'and common attention takes rope by default.')
    group.add_argument('--cross-entropy-loss-fusion', action='store_true',
                       help='Enabled fusion of cross entropy loss calculation.',
                       dest='cross_entropy_loss_fusion')
    group.add_argument('--cross-entropy-fusion-impl', type=str, default='native',
                       choices=['native', 'te'],
                       help='Implementation of cross entropy loss calculation.')
    group.add_argument('--use-flash-attn', action='store_true',
                       help='use FlashAttention implementation of attention. '
                       'https://arxiv.org/abs/2205.14135')
    group.add_argument('--disable-bias-linear', action='store_false',
                       help='Disable bias in the linear layers',
                       dest='add_bias_linear')
    group.add_argument('--add-qkv-bias', action='store_true',
                       help='Enable bias only in the QKV linear layers',
                       dest='add_qkv_bias')
    group.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'sgd'],
                       help='Optimizer function')
    group.add_argument('--optimizer-cpu-offload', action='store_true',
                       help='Offload optimizer state to CPU')
    group.add_argument('--optimizer-offload-fraction', type=float, default=1.0,
                          help='Ratio of optimizer state to offload to CPU')
    group.add_argument('--use-torch-optimizer-for-cpu-offload', action='store_true',
                       help="Use torch.optim.Optimizer instead of Megatron's optimizer in optimizer cpu offload mode.")
    group.add_argument('--overlap-cpu-optimizer-d2h-h2d', action='store_true', default=False,
                       help='Overlap CPU optimizer step, gradients D2H and updated parameters H2D.')
    group.add_argument('--dump-param-to-param-group-map', type=str, default=None,
                        help="Path to a file containing parameter-to-parameter-group mapping. "
                        "Provide a JSON file that specifies which parameters belong to which "
                        "parameter group for global coordination.")
    group.add_argument('--no-pin-cpu-grads', action='store_false', dest='pin_cpu_grads',
                       help='Disable pinning of CPU memory for gradients.')
    group.add_argument('--no-pin-cpu-params', action='store_false', dest='pin_cpu_params',
                       help='Disable pinning of CPU memory for parameters.')
    group.add_argument('--dataloader-type', type=str, default=None,
                       choices=['single', 'cyclic', 'external'],
                       help='Single pass vs multiple pass data loader')
    group.add_argument('--no-async-tensor-model-parallel-allreduce',
                       action='store_false',
                       help='DEPRECATED. This flag is ignored.',
                       dest='async_tensor_model_parallel_allreduce')
    group.add_argument('--no-persist-layer-norm', action='store_true',
                       help='Disable using persistent fused layer norm kernel. '
                       'This kernel supports only a set of hidden sizes. Please '
                       'check persist_ln_hidden_sizes if your hidden '
                       'size is supported.')
    group.add_argument('--sequence-parallel', action='store_true',
                       help='Enable sequence parallel optimization.')
    group.add_argument('--no-gradient-accumulation-fusion',
                       action='store_false',
                       help='Disable fusing gradient accumulation to weight '
                       'gradient computation of linear layers',
                       dest='gradient_accumulation_fusion')
    group.add_argument('--use-mcore-models', action='store_true',
                       dest='deprecated_use_mcore_models',
                       help='DEPRECATED. Use the implementation from megatron core.'
                       'Now ignored and mcore models are the default, use '
                       '--use-legacy-models to not use core models.')
    group.add_argument('--use-legacy-models', action='store_true',
                       help='Use the legacy Megatron models, not Megatron-Core models.')
    group.add_argument('--manual-gc', action='store_true',
                       help='Disable the threshold-based default garbage '
                       'collector and trigger the garbage collection manually. '
                       'Manual garbage collection helps to align the timing of '
                       'the collection across ranks which mitigates the impact '
                       'of CPU-associated jitters. When the manual gc is enabled, '
                       'garbage collection is performed only at the start and the '
                       'end of the validation routine by default.')
    group.add_argument('--manual-gc-interval', type=int, default=0,
                       help='Training step interval to trigger manual garbage '
                       'collection. When the value is set to 0, garbage '
                       'collection is not triggered between training steps.')
    group.add_argument('--no-manual-gc-eval', action='store_false',
                       help='When using manual garbage collection, disable '
                       'garbage collection at the start and the end of each '
                       'evaluation run.', dest='manual_gc_eval')
    group.add_argument('--disable-tp-comm-split-ag', action='store_false',
                       help='Disables the All-Gather overlap with fprop GEMM.',
                       dest='tp_comm_split_ag')
    group.add_argument('--disable-tp-comm-split-rs', action='store_false',
                       help='Disables the Reduce-Scatter overlap with fprop GEMM.',
                       dest='tp_comm_split_rs')
    group.add_argument('--pipeline-model-parallel-comm-backend', type=str, default=None,
                       choices=['nccl', 'ucc'],
                       help='Select a communicator backend for pipeline parallel communication. '
                       'If None, the default backend will be used.')
    group.add_argument('--high-priority-stream-groups', nargs='*', type=str, default=[],
                       help='The communicator group names to use high priority streams.')
    group.add_argument('--use-te-activation-func', action='store_true',
                       help='Use activation function kernel from Transformer Engine in MLP module.')

    return parser


def _add_rerun_machine_args(parser):
    group = parser.add_argument_group(title='rerun engine')

    group.add_argument('--error-injection-rate', type=int, default=0,
                       help='Rate at which to inject unexpected results, '
                       'e.g. 1000 means once every 1000 result validations')
    group.add_argument('--error-injection-type', type=str, default='transient_error',
                       choices=['correct_result', 'transient_error', 'persistent_error'],
                       help='Type of error to inject. ')
    group.add_argument('--rerun-mode', type=str, default='validate_results',
                       choices=['disabled', 'validate_results', 'report_stats'],
                       help='Use re-run engine to validate results (default) '
                       'or to emit stats on variability of computations due to '
                       'non-deterministic algorithms.')

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
    group.add_argument('--embedding-init-method-std', type=float, default=None,
                       help='Standard deviation of the zero mean normal '
                       'distribution used for embedding weight initialization. '
                       'If unset, embeddings will be initialized the same way '
                       'as other weights. Setting this to a value around 1.0 '
                       'may avoid loss spikes in training. Setting this to any '
                       'value will also skip applying weight decay on embedding '
                       'weights to avoid shrinkage towards zero. See '
                       'https://arxiv.org/abs/2312.16903 for more details.'
                       )
    group.add_argument('--init-method-xavier-uniform', action='store_true',
                       help='Enable Xavier uniform parameter initialization')

    return parser


def _add_learning_rate_args(parser):
    group = parser.add_argument_group(title='learning rate')

    group.add_argument('--lr', type=float, default=None,
                       help='Initial learning rate. Depending on decay style '
                       'and initial warmup, the learning rate at each '
                       'iteration would be different.')
    group.add_argument('--lr-decay-style', type=str, default='linear',
                       choices=['constant', 'linear', 'cosine', 'inverse-square-root', 'WSD'],
                       help='Learning rate decay function.')
    group.add_argument('--lr-wsd-decay-style', type=str, default='exponential',
                       choices=['exponential', 'linear', 'cosine', 'minus_sqrt'],
                       help='Decay style for the annealing phase of WSD'),
    group.add_argument('--lr-decay-iters', type=int, default=None,
                       help='number of iterations to decay learning rate over,'
                       ' If None defaults to `--train-iters`')
    group.add_argument('--lr-decay-samples', type=int, default=None,
                       help='number of samples to decay learning rate over,'
                       ' If None defaults to `--train-samples`')
    group.add_argument('--lr-wsd-decay-samples', type=int, default=None,
                       help='number of samples for the annealing phase in the wsd schedule')
    group.add_argument('--lr-wsd-decay-iters', type=int, default=None,
                       help='number of iterations for the annealing phase in the wsd schedule')
    group.add_argument('--lr-warmup-fraction', type=float, default=None,
                       help='fraction of lr-warmup-(iters/samples) to use '
                       'for warmup (as a float)')
    group.add_argument('--lr-warmup-iters', type=int, default=0,
                       help='number of iterations to linearly warmup '
                       'learning rate over.')
    group.add_argument('--lr-warmup-samples', type=int, default=0,
                       help='number of samples to linearly warmup '
                       'learning rate over.')
    group.add_argument('--lr-warmup-init', type=float, default=0.0,
                       help='Initial value for learning rate warmup. The '
                       'scheduler starts warmup from this value.')
    group.add_argument('--warmup', type=int, default=None,
                       help='Old lr warmup argument, do not use. Use one of the'
                       '--lr-warmup-* arguments above')
    group.add_argument('--min-lr', type=float, default=0.0,
                       help='Minimum value for learning rate. The scheduler'
                       'clip values below this threshold.')
    group.add_argument('--override-opt_param-scheduler', '--override-opt-param-scheduler',
                       action='store_true',
                       help='Reset the values of the scheduler (learning rate,'
                       'warmup iterations, minimum learning rate, maximum '
                       'number of iterations, and decay style from input '
                       'arguments and ignore values from checkpoints. Note'
                       'that all the above values will be reset.')
    group.add_argument('--use-checkpoint-opt_param-scheduler', '--use-checkpoint-opt-param-scheduler',
                       action='store_true',
                       help='Use checkpoint to set the values of the scheduler '
                       '(learning rate, warmup iterations, minimum learning '
                       'rate, maximum number of iterations, and decay style '
                       'from checkpoint and ignore input arguments.')
    group.add_argument('--decoupled-lr', type=float, default=None,
                       help='Separate learning rate for the input and output layer')
    group.add_argument('--decoupled-min-lr', type=float, default=None,
                       help='Minimum value for learning rate for the input and output layer. The scheduler'
                       'clip values below this threshold')

    return parser


def _add_checkpointing_args(parser):
    group = parser.add_argument_group(title='checkpointing')

    group.add_argument('--save', type=str, default=None,
                       help='Output directory to save checkpoints to.')
    group.add_argument('--save-interval', '--persistent-save-interval', type=int, default=None,
                       help='Number of iterations between persistent checkpoint saves.')
    group.add_argument('--save-retain-interval', type=int, default=None,
                       help='Number of iterations between retained checkpoints (other'
                       'checkpoints _except the last checkpoint_ are automatically deleted).')
    group.add_argument('--no-save-optim', action='store_true', default=None,
                       help='Do not save current optimizer.')
    group.add_argument('--no-save-rng', action='store_true', default=None,
                       help='Do not save current rng state.')
    group.add_argument('--load', type=str, default=None,
                       help='Directory containing a model checkpoint.')
    group.add_argument('--no-load-optim', action='store_true', default=None,
                       help='Do not load optimizer when loading checkpoint.')
    group.add_argument('--load-main-params-from-ckpt', action='store_true', default=None,
                       help='Load main parameters from checkpoint directly.')
    group.add_argument('--no-load-rng', action='store_true', default=None,
                       help='Do not load rng state when loading checkpoint.')
    group.add_argument('--no-strict-fsdp-dtensor-load', action='store_false', dest='strict_fsdp_dtensor_load',
                       help='Do not strict loading for fsdp_dtensor checkpoint format.')
    group.add_argument('--non-persistent-save-interval', type=int, default=None,
                       help='Number of iterations between non-persistent saves.')
    group.add_argument('--non-persistent-ckpt-type', type=str, default=None,
                       choices=['global', 'local', 'in_memory', None],
                       help='Type of non-persistent model checkpoints. '
                           '"global" - Saved as a standard checkpoint (e.g., on Lustre) with old checkpoints being removed. '
                           '"local" - Each rank saves a portion of the checkpoint locally (e.g., on SSD/ramdisk). '
                           'None - No non-persistent checkpointing (default option).')
    group.add_argument('--non-persistent-global-ckpt-dir', type=str, default=None,
                       help='Directory containing global non-persistent model checkpoints.')
    group.add_argument('--non-persistent-local-ckpt-dir', type=str, default=None,
                       help='Directory containing local non-persistent model checkpoints.')
    group.add_argument('--non-persistent-local-ckpt-algo', type=str, default='fully_parallel',
                       choices=['fully_parallel', 'atomic'],
                       help='Algorithm for local non-persistent checkpointing.')
    group.add_argument('--finetune', action='store_true',
                       help='Load model for finetuning. Do not load optimizer '
                       'or rng state from checkpoint and set iteration to 0. '
                       'Assumed when loading a release checkpoint.')
    group.add_argument('--pretrained-checkpoint', type=str, default=None,
                       help='Directory containing a pretrained model checkpoint for finetuning.')
    group.add_argument('--ckpt-step', type=int, default=None,
                       help='Checkpoint step to load model from.')
    group.add_argument('--no-initialization', action='store_false',
                       help='Do not perform initialization when building model, '
                       'can reduce startup time when definitely loading from a '
                       'checkpoint',
                       dest='perform_initialization')
    group.add_argument('--use-checkpoint-args', action='store_true',
                       help='Override model-related command-line arguments with arguments from checkpoint')
    group.add_argument('--use-mp-args-from-checkpoint-args', action='store_true',
                       help='Copy model parallelism command-line arguments from checkpoint')
    group.add_argument('--no-use-tokenizer-model-from-checkpoint-args', action='store_false',
                       dest='use_tokenizer_model_from_checkpoint_args',
                       help='If set, do not use tokenizer model path from checkpoint')
    group.add_argument('--exit-on-missing-checkpoint', action='store_true',
                       help="If '--load' is set, but checkpoint is not found "
                       "(e.g., path typo), then exit instead of random "
                       "initialization.")
    group.add_argument('--use-dist-ckpt', action='store_true',
                       dest='use_dist_ckpt_deprecated',
                       help='Deprecated: see --ckpt-format.')
    group.add_argument('--use-persistent-ckpt-worker', action='store_true',
                       help='Enables a persitent checkpoint worker for async save')

    group.add_argument('--auto-detect-ckpt-format', action='store_true',
                       help='Determine if the checkpoint format is in legacy or distributed format.'
                            ' If False, expects distributed checkpoint iff args.ckpt_format != "torch".'
                            ' Might slow down loading a bit (double rank0 ckpt load).')
    group.add_argument('--dist-ckpt-format',
                       dest='dist_ckpt_format_deprecated',
                       help='Deprecated: see --ckpt-format.')
    group.add_argument('--ckpt-format', default='torch_dist',
                       choices=['torch', 'torch_dist', 'zarr', 'torch_dcp', 'fsdp_dtensor'],
                       help='Checkpoint format to use. torch is the format used by torch.save/load.'
                       ' torch_dist is a megatron built-in distributed checkpointing format.'
                       ' torch_dcp is the torch.distributed.checkpoint format.'
                       ' fsdp_dtensor is a torch DCP native, Megatron FSDP training-specific checkpoint format.')
    group.add_argument('--ckpt-convert-format', default=None,
                       choices=['torch', 'torch_dist', 'zarr'],
                       help='Checkpoint format for conversion.')
    group.add_argument('--ckpt-convert-save', default=None,
                       help='Save directory for converted checkpoint.')
    group.add_argument('--ckpt-convert-update-legacy-dist-opt-format', action='store_true',
                       help='When loading a checkpoint, update the legacy format '
                       'for the distributed optimizer, which previously used a '
                       'merged param/grad buffer and a different bucket mapping. '
                       'The legacy format was deprecated on Feb 13, 2024.')
    group.add_argument('--ckpt-fully-parallel-save', action='store_true',
                       dest='ckpt_fully_parallel_save_deprecated',
                       help='Deprecated: see --no-ckpt-fully-parallel-save.')
    group.add_argument('--no-ckpt-fully-parallel-save', action='store_false',
                       dest='ckpt_fully_parallel_save',
                       help='Disable applying full save parallelization across DP for'
                            ' distributed checkpoints. Depending on ckpt format'
                            ' might decrease the number of files in the checkpoint.'
                            ' Makes DistributedOptimizer checkpoint non-reshardable.')
    group.add_argument('--async-save', action='store_true', default=None,
                       help='Apply async checkpointing save. Currently works only with'
                            '`torch_dist` distributed checkpoint format.')
    group.add_argument('--ckpt-fully-parallel-load', action='store_true',
                       help='Apply full load parallelization across DP for'
                            ' distributed checkpoints.')
    group.add_argument('--ckpt-assume-constant-structure', action='store_true',
                       help='If the model and optimizer state dict structure is'
                            'constant throughout a *single training job*, it allows for'
                            'different checkpointing performance optimizations.')
    group.add_argument('--dist-ckpt-strictness', type=str, default='assume_ok_unexpected',
                       choices=[e.value for e in StrictHandling],
                       help='Determine handling of key mismatch during checkpoint load.'
                            ' Check StrictHandling docs for flags meaning.'
                            ' NOTE: This flag controls only distributed checkpoint'
                            ' load from storage, not loading state dict into the model.')
    group.add_argument('--dist-ckpt-save-pre-mcore-014', action='store_true',
                       help='Revert checkpointing simplifications introduced in Megatron-Core'
                            ' v0.14. This option affects only checkpoint saving format and will'
                            ' be removed soon (checkpoint load format is determined based on'
                            ' checkpoint metadata).')
    group.add_argument('--dist-ckpt-optim-fully-reshardable', action='store_true',
                       help='Make optimizer distributed checkpoint fully reshardable (TP/PP/EP/DP)'
                            ' as opposed to plain DP reshardability.')
    group.add_argument('--distrib-optim-fully-reshardable-mem-efficient', action='store_true',
                       help='During distributed optimizer checkpoint save and load tries to use as'
                            ' little memory as possible by using Gloo (instead of NCCL) and only one'
                            ' rank for saving. Turn on only if experiencing host or device memory'
                            ' issues. Has affect only with `--dist-ckpt-optim-fully-reshardable`'
                            ' flag.')
    return parser


def _add_mixed_precision_args(parser):
    group = parser.add_argument_group(title='mixed precision')

    group.add_argument('--fp16', action='store_true',
                       help='Run model in fp16 mode.')
    group.add_argument('--bf16', action='store_true',
                       help='Run model in bfloat16 mode.')
    group.add_argument('--grad-reduce-in-bf16', action='store_true',
                       help='Reduce gradients in bfloat16.')
    group.add_argument('--loss-scale', type=float, default=None,
                       help='Static loss scaling, positive power of 2 '
                       'values can improve fp16 convergence. If None, dynamic'
                       'loss scaling is used.')
    group.add_argument('--initial-loss-scale', type=float, default=2**32,
                       help='Initial loss-scale for dynamic loss scaling.')
    group.add_argument('--min-loss-scale', type=float, default=1.0,
                       help='Minimum loss scale for dynamic loss scaling.')
    group.add_argument('--loss-scale-window', type=float, default=1000,
                       help='Window over which to raise/lower dynamic scale.')
    group.add_argument('--hysteresis', type=int, default=2,
                       help='hysteresis for dynamic loss scaling')
    group.add_argument('--fp32-residual-connection', action='store_true',
                       help='Move residual connections to fp32.')
    group.add_argument('--apply-query-key-layer-scaling', action='store_true',
                       help='Scale Q * K^T by 1 / layer-number. '
                       'Useful for fp16 training. Also sets `attention_softmax_in_fp32` to True.')
    group.add_argument('--attention-softmax-in-fp32', action='store_true',
                       help='Run attention masking and softmax in fp32.')
    group.add_argument('--accumulate-allreduce-grads-in-fp32',
                       action='store_true',
                       help='Gradient accumulation and all-reduce in fp32.')
    group.add_argument('--fp16-lm-cross-entropy', action='store_true',
                       help='Move the cross entropy unreduced loss calculation'
                       'for lm head to fp16.')
    group.add_argument('--disable-bf16-reduced-precision-matmul', action='store_true',
                       help='If True, sets torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False to '
                       'prevent matmul from using reduced precision accumulation when using BF16.')
    group.add_argument('--reuse-grad-buf-for-mxfp8-param-ag', action='store_true',
                       help='If True, reuse the grad buffer for MXFP8 parameter all-gather.')

    return parser


def _add_distributed_args(parser):
    group = parser.add_argument_group(title='distributed')

    group.add_argument('--tensor-model-parallel-size', type=int, default=1,
                       help='Degree of tensor model parallelism.')
    group.add_argument('--pipeline-model-parallel-size', type=int, default=1,
                       help='Degree of pipeline model parallelism.')
    group.add_argument('--decoder-first-pipeline-num-layers',
                       type=int, default=None,
                       help=('The number of transformer layers on the first pipeline stage of the decoder. '
                       'Default None is even split of transformer layers across all pipeline stages'))
    group.add_argument('--decoder-last-pipeline-num-layers',
                       type=int, default=None,
                       help=('The number of transformer layers on the last pipeline stage of the decoder. '
                       'Default None is even split of transformer layers across all pipeline stages'))
    group.add_argument('--pipeline-model-parallel-layout',
                       type=str, default=None,
                       help=('A string that describes a custom pipeline model parallel layout. '
                       'e.g., "E|(t|)*3,m|m||L". E, L, t, m denotes embedding, loss, transformer '
                       'decoder layer, and mtp layer, respectively. Stages are split by "|". '
                       'Replicated stages or layers can be described with multiplication. '
                       'Commas can be used cosmetically. '
                       'Default None is not using this argument to set the layout.'))
    group.add_argument('--model-parallel-size', type=int, default=None,
                       help='Old model parallel argument, do not use. Use '
                       '--tensor-model-parallel-size instead.')
    group.add_argument('--num-layers-per-virtual-pipeline-stage', type=int, default=None,
                       help='Number of layers per virtual pipeline stage')
    group.add_argument('--num-virtual-stages-per-pipeline-rank', type=int, default=None,
                       help='Number of virtual pipeline stages per pipeline parallelism rank')
    group.add_argument('--microbatch-group-size-per-virtual-pipeline-stage', type=int, default=None,
                       help='Number of contiguous microbatches per virtual pipeline stage',
                       dest='microbatch_group_size_per_vp_stage')
    group.add_argument('--no-overlap-p2p-communication', action='store_false',
                       help='overlap pipeline parallel communication with forward and backward chunks in 1F1B',
                       dest='overlap_p2p_comm')
    group.add_argument('--overlap-p2p-communication-warmup-flush', action='store_true',
                       default=False, help='if set, overlap pipeline parallel communication in warmup and flush',
                       dest='overlap_p2p_comm_warmup_flush')
    group.add_argument('--distributed-backend', default='nccl',
                       choices=['nccl', 'gloo'],
                       help='Which backend to use for distributed training.')
    group.add_argument('--distributed-timeout-minutes', type=int, default=10,
                       help='Default timeout minutes for torch.distributed.')
    group.add_argument('--distributed-timeout-seconds-after-init', type=int, default=None,
                       help='Timeout seconds for process groups after initialization.'
                            'This timeout is applied to all process groups after initialization.')
    group.add_argument('--overlap-grad-reduce', action='store_true',
                       default=False, help='If set, overlap DDP grad reduce.')
    group.add_argument('--defer-embedding-wgrad-compute', action='store_true',
                       default=False, help='If set, defers the vocabulary projection linear layer weight'
                       'gradient compute to pipeline flush.', dest='defer_embedding_wgrad_compute')
    group.add_argument('--wgrad-deferral-limit', type=int, default=0, help='Number of micro-batches for which'
                       'weight gradient computation of vocabulary projection is deferred, defaults to 0 which'
                       'means all the micro-batches are deferred. Invalid if `defer-embedding-wgrad-compute`'
                       'is not set')
    group.add_argument('--no-align-grad-reduce', action='store_false',
                       help='If not set, all PP stages will launch gradient reduces simultaneously. '
                       'Otherwise, each PP stage will independently launch as needed.',
                       dest='align_grad_reduce')
    group.add_argument('--ddp-num-buckets', type=int, default=None,
                       help='Number of buckets for data-parallel communication')
    group.add_argument('--ddp-bucket-size', type=int, default=None,
                       help='Bucket size for data-parallel communication')
    group.add_argument('--ddp-pad-buckets-for-high-nccl-busbw', action='store_true',
                       default=False, help='If set, make sure the bucket size is divisible by a large power '
                       'of 2 (2^16) to ensure NCCL collectives have high bus bandwidth at large DP counts, '
                       'since NCCL message size (which for ring algorithms is bucket_size / dp_size) '
                       'apparently needs to be divisible by a power of 2 for high busbw.')
    group.add_argument('--ddp-average-in-collective', action='store_true',
                       default=False, help='If set, average directly in data-parallel communication collective.')
    group.add_argument('--overlap-param-gather', action='store_true',
                       default=False, help='If set, overlap param all-gather in distributed optimizer.')
    group.add_argument('--overlap-param-gather-with-optimizer-step', action='store_true',
                       default=False, help='If set, overlap param all-gather of first bucket with optimizer step.')
    group.add_argument('--no-align-param-gather', action='store_false',
                       help='If not set, all PP stages will launch param all-gathers simultaneously. '
                       'Otherwise, each PP stage will independently launch as needed.',
                       dest='align_param_gather')
    group.add_argument('--no-scatter-gather-tensors-in-pipeline', action='store_false',
                       help='If not set, use scatter/gather to optimize communication of tensors in pipeline.',
                       dest='scatter_gather_tensors_in_pipeline')
    group.add_argument('--use-ring-exchange-p2p', action='store_true',
                       default=False, help='If set, use custom-built ring exchange '
                       'for p2p communications. Note that this option will require '
                       'a custom built image that support ring-exchange p2p.')
    group.add_argument('--local-rank', type=int, default=int(os.getenv('LOCAL_RANK', '0')),
                       help='local rank passed from distributed launcher.')
    group.add_argument('--lazy-mpu-init', type=bool, required=False,
                       help='If set to True, initialize_megatron() '
                       'skips DDP initialization and returns function to '
                       'complete it instead. Also turns on '
                       '--use-cpu-initialization flag. This is for '
                       'external DDP manager.' )
    group.add_argument('--account-for-embedding-in-pipeline-split', action='store_true',
                       default=False, help='If set, *input* embedding layer will be treated as a standard transformer'
                       'layer in the context of partition and placement for pipeline parallelism.')
    group.add_argument('--account-for-loss-in-pipeline-split', action='store_true',
                       default=False, help='If set, loss layer will be treated as a standard transformer'
                       'layer in the context of partition and placement for pipeline parallelism.')
    group.add_argument('--use-distributed-optimizer', action='store_true',
                       help='Use distributed optimizer.')
    group.add_argument('--use-nccl-ub', action='store_true', dest='nccl_ub', 
                       help='Use the userbuffer registration for DP/FSDP communication buffers.'
                       'This option will reduce GPU SM usage for the DP/FSDP communication,'
                       'which is improving the performance of the overlapped computation.')
    group.add_argument('--disable-symmetric-registration', action='store_true', dest='disable_symmetric_registration',
                       default=False, help='Disable symmetric (window) registration for NCCL userbuffer registration.'
                       'This option will force to use conventional (local) userbuffer registration when use-nccl-ub is set.')
    group.add_argument('--use-sharp', action='store_true', 
                       help='Required to enable SHARP communication.')
    group.add_argument('--sharp-enabled-group', type=str, default=None,
                       choices=['dp', 'dp_replica'],
                       help='IB SHARP can be enabled from only one communication group. '
                       'By default, it is enabled from dp group. '
                       'Available options: [dp, dp_replica]')
    group.add_argument('--use-megatron-fsdp', action='store_true',
                       help='Use the Megatron FSDP code path in DDP.')
    group.add_argument('--init-model-with-meta-device', action='store_true')
    group.add_argument('--data-parallel-sharding-strategy', type=str, default='no_shard',
                       choices=['no_shard', 'optim', 'optim_grads', 'optim_grads_params'],
                       help='Sharding strategy of data parallelism.')
    group.add_argument('--no-gradient-reduce-div-fusion', action='store_false', dest='gradient_reduce_div_fusion',
                       help='If not set, fuse the division in gradient reduce.')
    group.add_argument('--fsdp-double-buffer', action='store_true',
                       help="Enable double buffering for temporary memory needed for Megatron FSDP communications. "
                        "Double-buffering the communication memory improves memory management efficiency by "
                        "reusing previously allocated buffers, rather than creating new buffers for each FSDP communication. "
                        "This is required for user buffer registration and is enabled by default when using NCCL user buffers.")
    group.add_argument('--suggested-communication-unit-size', type=int, default=None,
                   help='Specifies the number of elements to communicate at once during FSDP (Fully Sharded Data Parallel) operations. '
                        'This flag also affects FSDP all-gather prefetch behavior. Setting a larger value increases the communication buffer size, '
                        'while a smaller value disables prefetching and may degrade performance. Adjust this value based on your system\'s memory '
                        'and performance requirements.')
    group.add_argument('--keep-fp8-transpose-cache', action='store_true',
                       help='If set, keep the fp8 transpose cache when using Megatron FSDP.')
    group.add_argument('--enable-full-sharding-in-hsdp', action='store_true',
                       help='If set, enable full sharding in megatron-fsdp Hybrid Sharded Data Parallel (HSDP) mode.')
    group.add_argument('--num-distributed-optimizer-instances', type=int, default=1,
                       help='Number of Distributed Optimizer copies across Data Parallel domain.')
    group.add_argument('--use-torch-fsdp2', action='store_true',
                       help='Use the torch FSDP2 implementation. FSDP2 has not been tested with pipeline parallelism, '
                       'and may contain bugs.')
    group.add_argument('--torch-fsdp2-no-reshard-after-forward', action='store_false', dest='torch_fsdp2_reshard_after_forward',
                       help='Whether to reshard weights after forward pass when using PyTorch FSDP2. '
                       'Set to enable FSDP ZeRO-2.')
    group.add_argument('--context-parallel-size', type=int, default=1,
                       help='Degree of context parallelism.')
    group.add_argument('--cp-comm-type', nargs='+', type=str, default=["p2p"],
                       help='Inter-gpu communication type for context parallelism: '
                       'p2p, a2a, allgather or a2a+p2p. If a single string is provided, '
                       'all layers will share the same communication type. Users can also '
                       'specify separated types for each layer like '
                       '--cp-comm-type p2p p2p a2a a2a a2a+p2p a2a+p2p')
    group.add_argument('--hierarchical-context-parallel-sizes', nargs='+', type=int, default=None,
                       help='Degrees of the hierarchical context parallelism. Users should '
                       'provide a list to specify the sizes for different levels. '
                       '--hierarchical-context-parallel-sizes 2 4 indicates every two adjacent gpus '
                       'forms the first level of cp groups and the cp ranks with the same odevity '
                       'forms the second level of cp groups.')
    group.add_argument('--nccl-communicator-config-path', type=str, default=None,
                       help='Path to the yaml file with NCCL communicator '
                       'configurations. The number of min/max thread groups and thread '
                       'group cluster size of each communicator can be configured by '
                       'setting `min_ctas`, `max_ctas`, and `cga_cluster_size`.')
    group.add_argument('--use-tp-pp-dp-mapping', action='store_true', default=False,
                        help='If set, distributed ranks initialize order is changed '
                        'from tp-cp-ep-dp-pp to tp-cp-ep-pp-dp.')
    group.add_argument('--replication', action='store_true', default=False,
                       help="If set, replication of local checkpoints is enabled. "
                       "Needs to be enabled on all ranks.")
    group.add_argument('--replication-jump', default=None, type=int,
                       help="Specifies `J`, the spacing between ranks storing replicas of a given rank's data. "
                       "Replicas for rank `n` may be on ranks `n+J`, `n+2J`, ..., or `n-J`, `n-2J`, etc. "
                       "This flag has an effect only if --replication is used. "
                       "and must be consistent across all ranks.")
    group.add_argument('--replication-factor', default=2, type=int,
                       help="Number of machines storing the replica of a given rank's data.")
    return parser


def _add_validation_args(parser):
    group = parser.add_argument_group(title='validation')

    group.add_argument('--full-validation', action='store_true', help='If set, each time validation occurs it uses the full validation dataset(s). This currently only works for GPT datasets!')
    group.add_argument('--multiple-validation-sets', action='store_true', help='If set, multiple datasets listed in the validation split are evaluated independently with a separate loss for each dataset in the list. This argument requires that no weights are included in the list')
    group.add_argument('--eval-iters', type=int, default=100,
                       help='Number of iterations to run for evaluation'
                       'validation/test for.')
    group.add_argument('--eval-interval', type=int, default=1000,
                       help='Interval between running evaluation on '
                       'validation set.')
    group.add_argument("--test-mode", action="store_true", help='Run all real-time test alongside the experiment.')
    group.add_argument('--skip-train', action='store_true',
                       default=False, help='If set, bypass the training loop, '
                       'optionally do evaluation for validation/test, and exit.')

    return parser


def _add_tokenizer_args(parser):
    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--vocab-size', type=int, default=None,
                       help='Size of vocab before EOD or padding.')
    group.add_argument('--padded-vocab-size', type=int, default=None,
                       help='Vocabulary size of the model (padded to be divisible by '
                       'tensor model parallel size). If not provided, it will be '
                       'automatically calculated from vocab-size.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file.')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file.')
    group.add_argument('--vocab-extra-ids', type=int, default=0,
                       help='Number of additional vocabulary tokens. '
                            'They are used for span masking in the T5 model')
    group.add_argument('--tokenizer-type', type=str,
                       default=None,
                       choices=['BertWordPieceLowerCase',
                                'BertWordPieceCase',
                                'GPT2BPETokenizer',
                                'SentencePieceTokenizer',
                                'GPTSentencePieceTokenizer',
                                'HuggingFaceTokenizer',
                                'Llama2Tokenizer',
                                'TikTokenizer',
                                'MultimodalTokenizer',
                                'NullTokenizer',
                                'NullMultimodalTokenizer',
                                'SFTTokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument('--tokenizer-model', type=str, default=None,
                       help='Sentencepiece tokenizer model.')
    group.add_argument('--tokenizer-metadata', type=str, default=None,
                       help='Path to tokenizer metadata in json format.')
    group.add_argument('--tiktoken-pattern', type=str, default=None,
                       help='Which tiktoken pattern to use. Options: [v1, v2]')
    group.add_argument('--tiktoken-num-special-tokens', type=int, default=1000,
                       help='Number of special tokens in tiktoken tokenizer')
    group.add_argument('--tiktoken-special-tokens', type=str, nargs='+', default=None,
                       help='List of tiktoken special tokens, needs to have '
                            '["<unk>", "<s>", "</s>", "<mask>", "<pad>", "<cls>", "<sep>"]')
    group.add_argument('--legacy-tokenizer', action='store_true', default=False,
                       help='To use legacy tokenizer system.')
    group.add_argument("--trust-remote-code", action="store_true",
                       help='Whether or not to allow PreTrainedTokenizer to execute remote code')
    return parser


def _add_data_args(parser):
    group = parser.add_argument_group(title='data and dataloader')

    group.add_argument('--data-path', nargs='*', default=None,
                       help='The weight and prefix list for a set of train, validation, and test'
                       'datasets which split according to --split. The accepted formats are: '
                       '(1) a single prefix, '
                       '(2) a list of weight prefix pairs e.g. weight1 prefix1 weight2 prefix2, '
                       '(3) a list of prefixes e.g. prefix1 prefix2. '
                       'For (3), weights are inferred from the lengths of the contributing datasets. '
                       'This argument is exclusive to the other independent --*-data-path arguments.')
    group.add_argument('--split', type=str, default=None,
                       help='Comma-separated list of proportions for training,'
                       ' validation, and test split. For example the split '
                       '`90,5,5` will use 90%% of data for training, 5%% for '
                       'validation and 5%% for test.')
    group.add_argument('--train-data-path', nargs='*', default=None,
                       help='The weight and prefix list for an independent train dataset. '
                       'Follows the same pattern rules as --data-path.')
    group.add_argument('--valid-data-path', nargs='*', default=None,
                       help='The weight and prefix list for an independent validation dataset. '
                       'Follows the same pattern rules as --data-path.')
    group.add_argument('--test-data-path', nargs='*', default=None,
                       help='The weight and prefix list for an independent test dataset. '
                       'Follows the same pattern rules as --data-path.')
    group.add_argument('--data-args-path', type=str, default=None,
                       help='Path to data-args. Instead of feeding `--data-path` '
                       'with weighted dataset, we pass in a file path from which '
                       'we read that argument. This is useful when the list of data is '
                       'too big.')
    group.add_argument('--per-split-data-args-path', type=str, default=None,
                       help='Path to per-split-data-args. Instead of feeding '
                       '`--(train|valid|test)-data-path` with weighted dataset, '
                       'we pass in a file path from which we read those arguments. '
                       'This is useful when the list of data is too big. Format is a '
                       'json file with `train`, `valid, `test` keys')
    group.add_argument('--data-cache-path', default=None,
                       help='Path to a directory to hold cached index files.')
    group.add_argument('--no-mmap-bin-files', action='store_false',
                       help='Disable mmap-ing of .bin files.',
                       dest='mmap_bin_files')
    group.add_argument('--mock-data', action='store_true',
                       help='Skip data loading and validation and opt for artificial '
                       'generation of mock data when an implementation is available.')
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
    group.add_argument('--num-workers', type=int, default=2,
                       help="Dataloader number of workers.")
    group.add_argument('--reset-position-ids', action='store_true',
                       help='Reset posistion ids after end-of-document token.')
    group.add_argument('--reset-attention-mask', action='store_true',
                       help='Reset self attention mask after '
                       'end-of-document token.')
    group.add_argument('--eod-mask-loss', action='store_true',
                       help='Mask loss for the end of document tokens.')
    group.add_argument('--no-create-attention-mask-in-dataloader', action='store_false',
                       help='If set, do not create attention_masks in dataloader.',
                       dest='create_attention_mask_in_dataloader')
    group.add_argument('--num-dataset-builder-threads', type=int, default=1,
                       help='Number of parallel threads per rank for dataset builder')
    group.add_argument('--object-storage-cache-path', type=str, default=None,
                       help='Path to cache index files when using s3 or msc dataloader')
    group.add_argument('--mid-level-dataset-surplus', type=float, default=0.005,
                       help='The sample surplus to build for the mid-level datasets(s)')
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

    # regularization arguments
    group.add_argument('--qk-layernorm', action='store_true',
                       help='Whether to layer normalize the q and k attention embeddings.')
    group.add_argument('--qk-l2-norm', action='store_true',
                       help='Use llama 4 qk l2 norm')

    return parser

def _add_moe_args(parser):
    group = parser.add_argument_group(title="moe")
    # General arguments
    group.add_argument('--expert-model-parallel-size', type=int, default=1,
                       help='Degree of expert model parallelism.')
    group.add_argument('--expert-tensor-parallel-size', type=int, default=None,
                       help='Degree of expert model parallelism. Default is None, which will be set to the value of --tensor-model-paralle-size.')
    group.add_argument('--num-experts', type=int, default=None,
                       help='Number of Experts in MoE (None means no MoE)')
    group.add_argument('--moe-layer-freq', type=moe_freq_type, default=1,
                       help='Frequency between MoE layers and Dense layers. Accepts either: '
                            '- An integer N: Represents a 1:N ratio, meaning one expert layer for every N-1 dense layers '
                            '- A string containing a Python list expression that defines a custom pattern, e.g.: '
                            '"([1]*3+[0]*1)*3" evaluates to [1,1,1,0,1,1,1,0,1,1,1,0] '
                            'where 1 indicates an expert layer and 0 indicates a dense layer. '
                            'Examples: "([0]+[1]*23)": 1 dense layer followed by 23 experts layers, '
                            '"([1]*3+[0]*2)*2": Three expert layers followed by two dense layers, repeated twice.')
    group.add_argument('--moe-ffn-hidden-size', type=int, default=None,
                       help='The hidden size of each expert\'s feed-forward network (ffn). '
                       'If not specified, defaults to the ffn_hidden_size.')
    group.add_argument('--moe-shared-expert-intermediate-size', type=int, default=None,
                       help='Shared expert total ffn hidden size. '
                       'It should be equal to "num_shared_experts * ffn_size_of_each_shared_expert" if there are multiple shared experts. '
                       'None means no shared expert. '
                       'By default, the shared experts execute before the router. However, when '
                       '--moe-shared-expert-overlap or --overlap-moe-expert-parallel-comm is set, '
                       'the shared experts execute after the router, before the routed experts. '
                       'This makes the gradients from the router and the shared experts added in '
                       'different orders to the hidden_states, causing minor numerical differences '
                       'in the hidden_states gradient.')
    group.add_argument('--moe-shared-expert-overlap', action='store_true',
                       help='Enable overlapping between shared expert computations and dispatcher communications. '
                       'Without this, the shared experts execute before the router. '
                       'Only effective when moe-shared-expert-intermediate-size is set.')
    group.add_argument('--moe-grouped-gemm', action='store_true',
                       help='When there are multiple experts per rank, launch multiple local GEMM kernels in multiple streams to improve the utilization and performance with GroupedLinear in TransformerEngine.')
    group.add_argument('--moe-use-legacy-grouped-gemm', action='store_true',
                       help='Use legacy GroupedMLP rather than TEGroupedMLP. Note: The legacy one will be deprecated soon.')
    group.add_argument('--moe-layer-recompute', action='store_true',
                       help='Enable checkpointing for moe_layer, should be used when memory is not sufficient. '
                       'Deprecated. Use "--recompute-granularity selective --recompute-modules moe" instead.')
    group.add_argument('--moe-extended-tp', action='store_true',
                       help='Deprecated. Use --expert-tensor-parallel-size instead.')
    group.add_argument('--moe-use-upcycling', action='store_true',
                       help='Load a checkpoint of a dense model, convert it into an MoE model, and save the converted model to the path specified by --save. '
                       'Upcycling is implemented on the top of distributed checkpointing, so it supports parallel modes different from the dense model.')
    # Router arguments
    group.add_argument('--moe-router-load-balancing-type', nargs='+', type=str,
                       choices=['aux_loss', 'seq_aux_loss', 'global_aux_loss', 'sinkhorn', 'none'],
                       default='aux_loss',
                       help='Determines the load balancing strategy for the router. "aux_loss" corresponds to the load balancing loss used in GShard and SwitchTransformer; "seq_aux_loss" corresponds to the load balancing loss used in DeepSeekV2, which computes the loss for each individual sample; "sinkhorn" corresponds to the balancing algorithm used in S-BASE, and "none" implies no load balancing. The default is "aux_loss".')
    group.add_argument('--moe-router-dtype', type=str,
                       choices=['fp32', 'fp64'],
                       default=None,
                       help='Data type for routing computation and expert output weighted averaging. '
                            'Fp32/fp64 enhances numerical stability, especially with numerous experts. '
                            'The perf impact should be negligible when used with permute fusion. '
                            'None means no changes for dtype.')
    group.add_argument('--moe-router-fusion', action='store_true',
                       help='Enable fusion for MoE TopK routing and aux-loss computation. This is only supported in TransformerEngine 2.7.0 and above.')
    group.add_argument('--moe-router-score-function', type=str,
                       choices=['softmax', 'sigmoid'],
                       default='softmax',
                       help='Score function for MoE TopK routing. Can be "softmax" or "sigmoid".')
    group.add_argument('--moe-router-topk', type=int, default=2,
                       help='Number of experts to route to for each token. The default is 2.')
    group.add_argument('--moe-router-pre-softmax', action='store_true',
                       help='Enable pre-softmax routing for MoE, which means softmax is before the top-k selection. By default, softmax is done after top-k.')
    group.add_argument('--moe-router-num-groups', type=int, default=None,
                       help='Number of groups to divide experts into for group-limited routing. When using group-limited routing: 1) Experts are divided into equal-sized groups, 2) For each token, a subset of groups are selected based on routing scores (sum of top-2 expert scores within each group), 3) From these selected groups, moe_router_topk experts are chosen.'
                       'Two common use cases: 1) Device-limited routing: Set equal to expert parallel size (EP) to limit each token to experts on a subset of devices (See DeepSeek-V2: https://arxiv.org/pdf/2405.04434) 2) Node-limited routing: Set equal to number of nodes in EP group to limit each token to experts on a subset of nodes (See DeepSeek-V3: https://arxiv.org/pdf/2412.19437)')
    group.add_argument('--moe-router-group-topk', type=int, default=None,
                       help='Number of selected groups for group-limited routing.')
    group.add_argument('--moe-router-topk-scaling-factor', type=float, default=None,
                       help='Scaling factor for routing score in top-k selection, only works when --moe-router-pre-softmax enabled. Defaults to None, which means no scaling.')
    group.add_argument('--moe-router-enable-expert-bias', action='store_true',
                       help='TopK routing with dynamic expert bias in the aux-loss-free load balancing strategy. '
                       'The routing decision is based on the sum of the routing scores and the expert bias. '
                       'See https://arxiv.org/abs/2408.15664 for details.')
    group.add_argument('--moe-router-bias-update-rate', type=float, default=1e-3,
                       help='Expert bias update rate in the aux-loss-free load balancing strategy. '
                       'The expert bias is updated based on the number of assigned tokens to each expert in a global batch, '
                       'where the bias is increased for the experts with less assigned tokens and decreased for the experts with more assigned tokens. '
                       'The default value 1e-3 is same as that used in DeepSeekV3.')
    group.add_argument('--moe-router-force-load-balancing', action='store_true',
                       help='[Experimental] Force override routing to balance token distribution using random logits for MoE routers, supporting naive top-k and group-limited top-k. This experimental feature is for benchmarking purposes only!')
    group.add_argument('--moe-router-padding-for-fp8', action='store_true',
                       help='Pad the routing_map to make sure the number of tokens each expert received '
                       'is a multiple of 16/32 for FP8 precision. It is suggested to enable this for '
                       'dropless training with FP8 precision when num_local_experts > 1. This is a more '
                       'efficient way to pad for FP8 which eliminates the explicit padding in the '
                       'GroupedMLP layer.')
    group.add_argument('--moe-aux-loss-coeff', type=float, nargs='+', default=0.0,
                       help='Scaling coefficient for the aux loss: a starting value of 1e-2 is recommended.')
    group.add_argument('--moe-z-loss-coeff', type=float, default=None,
                       help='Scaling coefficient for the z-loss: a starting value of 1e-3 is recommended.')
    group.add_argument('--moe-input-jitter-eps', type=float, default=None,
                       help='Add noise to the input tensor by applying jitter with a specified epsilon value.')
    group.add_argument('--moe-per-layer-logging', action='store_true',
                       help='Enable per-layer logging for MoE, currently supports auxiliary loss and z loss.')
    # Token dispatcher arguments
    group.add_argument('--moe-token-dispatcher-type', type=str,
                       choices=['allgather', 'alltoall', 'flex'],
                       default='allgather',
                       help="The type of token dispatcher to use. The default is 'allgather'. Options are 'allgather', 'alltoall'. We recommend using 'alltoall' when applying expert parallelism. For more information, please refer to the documentation in core/moe/README.")
    group.add_argument('--moe-enable-deepep', action='store_true',
                       help='[Experimental] Enable DeepSeek/DeepEP for efficient token dispatching and combine in MoE models. Only works with flex token dispatcher by setting --moe-token-dispatcher-type=flex.')
    group.add_argument('--moe-deepep-num-sms', type=int, default=20,
                       help='Number of SMs to use for DeepEP.')
    group.add_argument('--moe-permute-fusion', action='store_true',
                       help='Fuse token rearrangement ops during token dispatching.')
    # Token dropping arguments
    group.add_argument('--moe-expert-capacity-factor', type=float, default=None,
                       help='The capacity factor for each expert, None means no token will be dropped.')
    group.add_argument('--moe-pad-expert-input-to-capacity', action='store_true',
                       help='Pads the input for each expert to match the expert capacity length, effective only after the --moe-expert-capacity-factor is set.')
    group.add_argument('--moe-token-drop-policy', type=str, default='probs', choices=['probs', 'position'],
                       help='The policy to drop tokens. Can be either "probs" or "position". If "probs", the tokens with the lowest probabilities will be dropped. If "position", tokens at the end of each batch will be dropped.')
    group.add_argument('--moe-apply-probs-on-input', action='store_true',
                       help='Apply probs before mlp activation for moe routing.')
    # MoE communication overlap arguments
    group.add_argument('--overlap-moe-expert-parallel-comm', action='store_true',
                       help='Overlap the EP A2A communication by batch-level overlapping in 1f1b stage.')
    group.add_argument('--delay-wgrad-compute', action='store_true',
                       help='Delay the wgrad compute for batch-level overlapping')

    group.add_argument('--moe-upcycling-granularity', type=int, default=1,
                       help='This param sepecifics how many times smaller is the expert hidden size compared with the original dense FFN hidden size. '
                       'For using granular upcycling strategy, please set this param as a positive integer. If this param is set to 1, it means using the default upcycling strategy.')
    group.add_argument('--moe-pad-experts-for-cuda-graph-inference', action='store_true',
                       help="some MoE routers have a D2H sync that will break cuda graphs.  If this flag is set the router will switch" \
                       " to dropping and padding during decode time which does not have a D2H sync. The capacity factor is set to the" \
                       " max that an expert could see during inference so no tokens are actually dropped.")
    return parser

def _add_mla_args(parser):
    group = parser.add_argument_group(title="mla")
    group.add_argument('--q-lora-rank', type=int, default=None,
                       help="Rank of Query tensor's low rank representation.")
    group.add_argument('--kv-lora-rank', type=int, default=32,
                       help="Rank of Key and Value tensors' low rank representation.")
    group.add_argument('--qk-head-dim', type=int, default=128,
                       help="Dimension of the head in the QK projection. q_head_dim = qk_head_dim + qk_pos_emb_head_dim")
    group.add_argument('--qk-pos-emb-head-dim', type=int, default=64,
                       help="Dimension of the position embedding in the QK projection.")
    group.add_argument('--v-head-dim', type=int, default=128,
                       help="Dimension of the head in the V projection.")
    group.add_argument('--rotary-scaling-factor', type=float, default=1.0,
                       help="Rotary scaling factor for the rotary embeddings.")
    group.add_argument('--mscale', type=float, default=1.0,
                       help="Mscale for YaRN RoPE in multi-latent attention.")
    group.add_argument('--mscale-all-dim', type=float, default=0.0,
                       help="Mscale all dimensions for YaRN RoPE in multi-latent attention.")
    group.add_argument('--cache-mla-latents', action='store_true', default=False,
                       help="If set caches the mla down projected latents with mla flash decode.")

    return parser

def _add_heterogeneous_args(parser):
    """
    Heterogeneous models refer to transformer architectures where individual layers can differ 
    in configuration. Specifically:
        - Attention or MLP layers can be replaced with either a linear layer or a no-op 
        - MLP intermediate dimensions can vary between layers
    We use the format of the HuggingFace config files in llama nemotron models to define the architecture.
    For example, https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1/resolve/main/config.json

    Most notably, the "block_config" maps to a list of attention and mlp configurations for each layer.
    For example, the "block_config" for a 2 layer model is:
     "block_configs": [
        {
            "attention": {
                "n_heads_in_group": 8,
                "no_op": false,
                "replace_with_linear": false,
            },
            "ffn": {
                "ffn_mult": 2.625,
                "no_op": false,
                "replace_with_linear": false,
            }
        },
        {
            "attention": {
                "n_heads_in_group": null,
                "no_op": true,
                "replace_with_linear": false,
            },
            "ffn": {
                "ffn_mult": 2.625,
                "no_op": false,
                "replace_with_linear": false,
            }
        }
    ]
    """
    group = parser.add_argument_group(title="heterogeneous architecture")
    group.add_argument('--heterogeneous-layers-config-path', type=str, default=None,
                       help='Path to json file containing heterogeneous model configuration. '
                       'Use the format of the HuggingFace config files in llama nemotron '
                       'models, e.g. https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1/resolve/main/config.json.')
    group.add_argument('--heterogeneous-layers-config-encoded-json', type=str, default=None,
                       help='This is encoded json string of the heterogeneous model configuration. Used to keep the content '
                       'of the heterogeneous model specification in args when the model is loaded from a checkpoint. '
                       'Use the format of the HuggingFace config files in llama nemotron '
                       'models, e.g. https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1/resolve/main/config.json.')
    return parser

def _add_experimental_args(parser):
    group = parser.add_argument_group(title='experimental')

    group.add_argument('--enable-experimental', action='store_true',
                       help='Enable experimental features.')
    group.add_argument('--spec', type=str, default=None, nargs='*',
                       help='Specify the <module_location function_name> pair '
                       'that returns a spec to customize a model, transformer '
                       'block, or transformer layer, depending on the use case.'
                       'To use local spec specify local as the argument.'
                       'For more details, see the model class, '
                       '`transformer_block.py`, or `transformer_layer.py`')
    group.add_argument('--hybrid-attention-ratio', type=float, default=0.0,
                       help='Ratio of attention layers to total layers, in the '
                       'range [0.0, 1.0].')
    group.add_argument('--hybrid-mlp-ratio', type=float, default=0.0,
                       help='Ratio of mlp layers to total layers, in the '
                       'range [0.0, 1.0].')
    group.add_argument('--hybrid-override-pattern', type=str, default=None,
                       help='Force a specific hybrid layer pattern. The value'
                            'should be a string of characters chosen from'
                            'core.ssm.mamba_hybrid_layer_allocation.Symbols.'
                            'If a value greater than 0.0 is supplied to any of the '
                            'hybrid ratio arguments, then the number of each type'
                            'of layer in the override pattern must match number in'
                            'the overidden pattern')
    group.add_argument('--mamba-state-dim', type=int, default=128,
                       help='State dimension for Mamba layers.')
    group.add_argument('--mamba-head-dim', type=int, default=64,
                       help='Head dimension for Mamba layers.')
    group.add_argument('--mamba-num-groups', type=int, default=8,
                       help='Number of groups for Mamba layers.')
    group.add_argument('--mamba-num-heads', type=int, default=None,
                       help='Number of heads for Mamba layers.'
                       'If not set, then the number of heads will be '
                       '--hidden-size * expand // --mamba-head-dim')
    group.add_argument('--is-hybrid-model', default=False, action="store_true",
                       help='Indicates whether the model is a hybrid model.')
    group.add_argument('--disable-mamba-mem-eff-path', default=False, action="store_true",
                       help='Disable Mamba efficient path.')
    group.add_argument('--yaml-cfg', type=str, default=None,
                       help = 'Config file to add additional arguments')

    # Args of precision-aware optimizer
    group.add_argument('--use-precision-aware-optimizer', action='store_true',
                       help='Use the precision-aware optimizer in TransformerEngine, which allows '
                       'setting the main params and optimizer states to lower precision, such as '
                       'fp16, bf16 and fp8.')
    group.add_argument('--main-grads-dtype', default='fp32', choices=['fp32', 'bf16'],
                       help='Dtype of main grads when enabling precision-aware-optimizer')
    group.add_argument('--main-params-dtype', default='fp32', choices=['fp32', 'fp16'],
                       help='Dtype of main params when enabling precision-aware-optimizer')
    group.add_argument('--exp-avg-dtype', default='fp32', choices=['fp32', 'fp16', 'bf16', 'fp8'],
                       help='Dtype of exp_avg (1st moment in adam optimizer) when enabling '
                            'precision-aware-optimizer. This dtype is used for storing the '
                            'optimizer state in memory during training but does not affect '
                            'the precision in the kernel computation.')
    group.add_argument('--exp-avg-sq-dtype', default='fp32', choices=['fp32', 'fp16', 'bf16', 'fp8'],
                       help='Dtype of exp_avg_sq (2nd moment in adam optimizer) when enabling '
                            'precision-aware-optimizer. This dtype is used for storing the '
                            'optimizer state in memory during training but does not affect '
                            'the precision in the kernel computation.')
    return parser


def _add_msc_args(parser):
    group = parser.add_argument_group(title="msc")
    group.add_argument('--disable-msc', default=True, action='store_false', dest='enable_msc',
                       help='Disable the usage of Multi-Storage Client (MSC) in Megatron Core.')
    return parser

def _add_kitchen_quantization_arguments(parser: argparse.ArgumentParser):
    """Add quant-specific arguments to the main parser

    If kitchen isn't available, nothing to do here, return unchanged parser
    """
    try:
        from megatron.core.extensions.kitchen import KitchenSpecProvider

        have_kitchen = True
    except (ImportError, ModuleNotFoundError):
        have_kitchen = False

    if have_kitchen:
        group = parser.add_argument_group(title="kitchen")
        recipe_or_config_group = group.add_mutually_exclusive_group(required=False)
        recipe_or_config_group.add_argument(
            '--kitchen-config-file',
            type=str,
            default=None,
            help="Use the config .yaml file at the specified location to "
            "configure kitchen quantization.",
        )
        recipe_or_config_group.add_argument(
            '--kitchen-recipe-number',
            type=int,
            default=None,
            help="Use a default kitchen recipe for all layers as defined by QAT_PARAMS index",
        )
    return parser

def _add_sft_args(parser):
    group = parser.add_argument_group(title='sft')
    group.add_argument('--sft', action="store_true", help='Megatron SFT training')
    group.add_argument('--sft-tokenizer-prompt-format', type=str, default="nemotron-h-aligned", 
                       help='SFT prompt format.')
    return parser
