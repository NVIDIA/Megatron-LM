# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Megatron arguments."""

import argparse
import dataclasses
import json
import os
from pathlib import Path
import re
import types

import torch
import torch.nn.functional as F
from packaging.version import Version as PkgVersion

from megatron.core.dist_checkpointing.validation import StrictHandling
from megatron.core.rerun_state_machine import RerunStateMachine
from megatron.core.transformer import MLATransformerConfig, TransformerConfig
from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout
from megatron.core.transformer.enums import AttnBackend, CudaGraphScope
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

from megatron.training.argument_utils import ArgumentGroupFactory

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
    parser = _add_experimental_attention_variant_args(parser)
    parser = _add_heterogeneous_args(parser)
    parser = _add_logging_args(parser)
    parser = _add_straggler_detector_args(parser)
    parser = _add_workload_inspector_server_args(parser)
    parser = _add_inference_args(parser)
    parser = _add_transformer_engine_args(parser)
    parser = _add_experimental_args(parser)
    parser = _add_one_logger_args(parser)
    parser = _add_inprocess_restart_args(parser)
    parser = _add_ft_package_args(parser)
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
        warn_rank_0('The MSC feature is disabled.')

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
          "([0]+[1]*23)": 1 dense layer followed by 23 expert layers
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

def la_freq_type(x):
    """Frequency between LA (linear attention) layers and SDPA (scaled dot-product attention) layers.

    Accepts either:
    - An integer N: Represents a (N-1):N ratio, meaning (N-1) LA layers for every 1 SDPA layer
    - A string "N": Same as above, but provided as a string
    - A string containing a Python list expression that defines a custom pattern, e.g.:
      "([1]*3+[0]*1)*3" evaluates to [1,1,1,0,1,1,1,0,1,1,1,0]
      where 1 indicates an LA layer and 0 indicates a SDPA layer.
      This allows defining arbitrary patterns of LA and SDPA layers.
      The pattern length must match the total number of transformer layers.
      Examples:
          "([0]+[1]*23)": 1 SDPA layer followed by 23 LA layers
          "([1]*3+[0]*2)*2": Three LA layers followed by two SDPA layers, repeated twice.
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

    # Assert that `torch_memory_saver` is installed if offloading KV cache during RL.
    if args.rl_offload_kv_cache_during_training:
        try:
            from torch_memory_saver import torch_memory_saver
        except ImportError:
            raise AssertionError("To use offload-kv-cache-during-training, `torch_memory_saver` must be installed. See https://github.com/fzyzcjy/torch_memory_saver.")
        assert not args.inference_dynamic_batching_unified_memory_level, "The KV cache should not be instantiated in unified memory when it is offloaded during training."

    # Batch size checks if running RL.
    if args.perform_rl_step:
        assert not (args.rl_remove_kv_cache_during_training and args.rl_offload_kv_cache_during_training), \
            "Cannot use both remove-kv-cache-during-training and offload-kv-cache-during-training"

        assert not (args.rl_partial_rollouts and args.rl_remove_kv_cache_during_training), \
            "Cannot use both partial-rollouts and remove-kv-cache-during-training"

        # Validate inference model offloading - requires either UVM or torch_memory_saver
        if args.rl_offload_inference_model_weights_when_idle:
            if args.rl_inference_model_unified_memory_level != 1:
                # Not using UVM, so we need torch_memory_saver
                try:
                    from torch_memory_saver import torch_memory_saver
                except ImportError:
                    raise AssertionError(
                        "To use --rl-offload-inference-model-weights-when-idle without UVM "
                        "(--rl-inference-model-unified-memory-level=1), `torch_memory_saver` must be "
                        "installed. See https://github.com/fzyzcjy/torch_memory_saver."
                    )

        # When using different EP sizes for inference and training (EP refit), the legacy
        # GroupedMLP is not supported. Only SequentialMLP or TEGroupedMLP can be used.
        if (
            args.rl_inference_expert_model_parallel_size is not None
            and args.rl_inference_expert_model_parallel_size != args.expert_model_parallel_size
        ):
            assert not args.moe_use_legacy_grouped_gemm, (
                "Legacy GroupedMLP (--moe-use-legacy-grouped-gemm) is not supported when using "
                "different expert parallelism sizes for inference and training. "
                "Use SequentialMLP (default when --moe-grouped-gemm is not set) or "
                "TEGroupedMLP (--moe-grouped-gemm without --moe-use-legacy-grouped-gemm)."
            )

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
            assert args.micro_batch_size == 1, \
                "micro_batch_size must be 1 when using sequence packing. To increase compute per micro batch increase the sequence length."

    print_rank_0('using world size: {}, data-parallel size: {}, '
                 'context-parallel size: {}, '
                 'hierarchical context-parallel sizes: {}, '
                 'tensor-model-parallel size: {}, '
                 'pipeline-model-parallel size: {}'.format(
                     args.world_size, args.data_parallel_size,
                     args.context_parallel_size,
                     args.hierarchical_context_parallel_sizes,
                     args.tensor_model_parallel_size,
                     args.pipeline_model_parallel_size))

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
        print_rank_0('--checkpoint-activations is no longer valid, use --recompute-activations, '
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
            warn_rank_0('Overriding default arguments for {key}:{v} '
                        'with {key}:{v2}'.format(key=key, v=defaults[key],
                                                 v2=getattr(args, key)))
        else:
            setattr(args, key, defaults[key])

    if args.data_path is not None and args.split is None:
        legacy_default_split_value = '969, 30, 1'
        warn_rank_0('Please specify --split when using --data-path. Using legacy default value '
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

    if args.phase_transition_iterations:
        args.phase_transition_iterations = sorted(
            int(x.strip()) for x in args.phase_transition_iterations.split(",")
        )
        assert args.rampup_batch_size is None, "multi-phase training does not support batch size ramp-up"

    # Batch size.
    assert args.micro_batch_size is not None
    assert args.micro_batch_size > 0
    if args.global_batch_size is None:
        args.global_batch_size = args.micro_batch_size * args.data_parallel_size
        print_rank_0('setting global batch size to {}'.format(args.global_batch_size))
    assert args.global_batch_size > 0

    # === MTP validation ===
    # Deprecation warnings for legacy MTP arguments
    if args.mtp_hybrid_override_pattern is not None:
        warn_rank_0(
            "--mtp-hybrid-override-pattern is deprecated. "
            "For new hybrid models with MTP models, use unified --hybrid-override-pattern instead. "
            "Example: 'M*M*/MM/MM' means main='M*M*', MTP pattern='MM' with 2 depths. "
            "This argument is kept only for loading old checkpoints.",
            args.rank,
        )

    # Backward compatibility: convert legacy mtp_hybrid_override_pattern to unified format
    from megatron.core.ssm.mamba_hybrid_layer_allocation import Symbols, parse_hybrid_pattern
    sep = Symbols.MTP_SEPARATOR
    if (
        getattr(args, 'mtp_hybrid_override_pattern', None) is not None
        and args.mtp_num_layers is not None
        and args.mtp_num_layers > 0
        and (args.hybrid_override_pattern is None or sep not in args.hybrid_override_pattern)
    ):
        main_pattern = args.hybrid_override_pattern or ''
        mtp_pattern = args.mtp_hybrid_override_pattern
        args.hybrid_override_pattern = main_pattern + sep + sep.join([mtp_pattern] * args.mtp_num_layers)
        args.mtp_hybrid_override_pattern = None
        print_rank_0(f"Converted legacy MTP pattern to unified: {args.hybrid_override_pattern}")

    # Infer mtp_num_layers from unified pattern
    if args.hybrid_override_pattern and sep in args.hybrid_override_pattern:
        parsed = parse_hybrid_pattern(args.hybrid_override_pattern)
        if parsed.mtp_pattern and parsed.mtp_num_depths > 0:
            inferred_mtp_num_layers = parsed.mtp_num_depths
            if args.mtp_num_layers is None:
                args.mtp_num_layers = inferred_mtp_num_layers
            elif args.mtp_num_layers != inferred_mtp_num_layers:
                warn_rank_0(
                    f"--mtp-num-layers ({args.mtp_num_layers}) conflicts with "
                    f"MTP depth count ({inferred_mtp_num_layers}) in pattern '{args.hybrid_override_pattern}'. "
                    f"Using the inferred value ({inferred_mtp_num_layers}).",
                    args.rank
                )
                args.mtp_num_layers = inferred_mtp_num_layers

    # MTP validation
    if args.mtp_num_layers:
        assert not args.use_legacy_models, "The legacy Megatron models does not support Multi-Token Prediction (MTP)."
        assert args.position_embedding_type == "rope" or args.position_embedding_type == "none", (
            f"Multi-Token Prediction (MTP) is not supported with {args.position_embedding_type} position embedding type."
            + f"The supported position embedding types are rope and none."
        )

    # Validate MTP args for hybrid vs non-hybrid models
    if args.is_hybrid_model:
        # Mamba/hybrid model MTP validation
        if args.mtp_num_layers and not (args.hybrid_override_pattern and sep in args.hybrid_override_pattern):
            # Hybrid model wants MTP but no unified pattern - check for legacy args
            if args.mtp_hybrid_override_pattern is None:
                warn_rank_0(
                    "Hybrid model with --mtp-num-layers but no MTP pattern. "
                    "Use unified --hybrid-override-pattern with '/' separator (e.g., 'M*M*/MM/MM') "
                    "or legacy --mtp-hybrid-override-pattern for old checkpoints.",
                    args.rank
                )
    else:
        # Non-hybrid (GPT) model MTP validation
        if args.mtp_hybrid_override_pattern is not None:
            warn_rank_0(
                "--mtp-hybrid-override-pattern is for Mamba/hybrid models only. "
                "For GPT models, MTP replicates the main transformer layer structure. "
                "This argument will be ignored.",
                args.rank
            )
    # === End of MTP validation ===
    
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

    print_rank_0(
        f"Number of virtual stages per pipeline stage: {args.virtual_pipeline_model_parallel_size}"
    )

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
        
    if args.fsdp_manual_registration:
        assert args.use_megatron_fsdp, "FSDP manual registration is only supported with Megatron FSDP"
        assert args.nccl_ub, "FSDP manual registration is only supported with nccl-ub option"

        if args.use_megatron_fsdp:
            args.reuse_grad_buf_for_mxfp8_param_ag = False

    # Parameters dtype.
    args.params_dtype = torch.float
    if args.fp16:
        assert not args.bf16
        args.params_dtype = torch.half
        # Turn off checking for NaNs in loss and grads if using dynamic loss scaling,
        # where NaNs in grads / loss are signal to the loss scaler.
        if not args.loss_scale:
            args.check_for_nan_in_loss_and_grad = False
            warn_rank_0('Setting args.check_for_nan_in_loss_and_grad to False since '
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
            print_rank_0('accumulate and all-reduce gradients in fp32 for bfloat16 data type.')
    if args.cuda_graph_impl == "local" and CudaGraphScope.full_iteration in args.cuda_graph_scope:
        if not args.inference_dynamic_batching:
            assert not args.check_for_nan_in_loss_and_grad, \
            "--no-check-for-nan-in-loss-and-grad should be set with full_iteration CUDA graph"
        else:
            assert args.fp8 is None, \
            "fp8 is not supported with inference dynamic batching and full_iteration CUDA graph"

    print_rank_0('using {} for parameters ...'.format(args.params_dtype))

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
    if args.log_memory_interval is not None:
        assert args.log_memory_interval % args.log_interval == 0
    # Mixed precision checks.
    if args.fp16_lm_cross_entropy:
        assert args.fp16, 'lm cross entropy in fp16 only support in fp16 mode.'
    if args.fp32_residual_connection:
        assert args.fp16 or args.bf16, \
            'residual connection in fp32 only supported when using fp16 or bf16.'

    if args.moe_grouped_gemm:
        dc = torch.cuda.get_device_capability()
        assert dc[0] >= 8, "Unsupported compute capability for GroupedGEMM kernels."

    if args.no_weight_decay_cond_type is not None:
        print_rank_0(
            'WARNING: --no-weight-decay-cond-type is deprecated. Please use --apply-wd-to-qk-layernorm instead.',
            args.rank,
        )
        if args.no_weight_decay_cond_type == "apply_wd_to_qk_layernorm":
            args.apply_wd_to_qk_layernorm = True
        else:
            raise ValueError(f"Invalid no_weight_decay_cond_type: {args.no_weight_decay_cond_type}")
        args.no_weight_decay_cond_type = None

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
        print_rank_0('Persistent fused layer norm kernel is supported from '
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

    if args.hybrid_context_parallel:
        assert not args.pipeline_model_parallel_size > 1, 'Hybrid context parallelism not supported with pipeline parallelism'
        assert not args.enable_cuda_graph, 'Hybrid context parallelism not supported with CUDA Graph'
        assert not args.use_megatron_fsdp, 'Hybrid context parallelism not supported with Megatron FSDP'
        assert args.dataloader_type == 'single', 'Hybrid context parallelism only supported with single dataloader type'
        assert args.calculate_per_token_loss, 'Hybrid context parallelism must be used with --calculate-per-token-loss'

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

    if args.qk_clip:
        assert is_te_min_version("2.9.0"), \
            '--qk-clip is only supported with TE >= 2.9.0.'
        assert 0.0 < args.qk_clip_alpha < 1.0, \
            '--qk-clip-alpha must be between 0.0 and 1.0 when using --qk-clip.'
        assert args.qk_clip_threshold > 0, \
            '--qk-clip-threshold must be greater than 0 when using --qk-clip.'

    # decoupled log max attention logit check
    if args.log_max_attention_logit:
        assert is_te_min_version("2.9.0"), \
            '--log-max-attention-logit is only supported with TE >= 2.9.0.'

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
    if args.num_experts is not None and args.moe_ffn_hidden_size is None:
        args.moe_ffn_hidden_size = args.ffn_hidden_size
        warn_rank_0("moe_ffn_hidden_size is not set, using ffn_hidden_size for MoE instead.")

    # Context parallel
    if args.context_parallel_size > 1:
        assert not args.use_legacy_models, "Context parallelism is not supported in legacy models."

    # Expert parallelism check
    if args.expert_model_parallel_size  > 1:
        assert args.num_experts is not None, "num_experts must be non None to use expert model parallelism"
        assert args.num_experts % args.expert_model_parallel_size == 0, \
            "Number of experts should be a multiple of expert model parallel_size."

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

    if args.fim_data:
        extra_tokens = [
            args.fim_prefix_token,
            args.fim_middle_token,
            args.fim_suffix_token,
            args.fim_pad_token,
            args.fim_eod_token,
        ]
        assert not args.mock_data, "Mock dataset is not supported with FIM dataset."
        assert args.fim_rate, "--fim-rate should be specified."
        assert args.fim_spm_rate, "--fim-spm-rate should be specified."
        assert all(token is not None for token in extra_tokens), "FIM extra tokens should be specified."

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

    if args.fake_process_group:
        assert args.moe_token_dispatcher_type != "flex", "Fake process group is not supported with flex token dispatcher."
        # Disable nan check for fake process group
        args.check_for_nan_in_loss_and_grad = False
        warn_rank_0('check_for_nan_in_loss_and_grad is set to False for fake process group.')
        # Disable gloo process groups for fake process group
        args.enable_gloo_process_groups = False
        warn_rank_0('enable_gloo_process_groups is set to False for fake process group.')

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

    if args.use_dist_ckpt and args.async_save:
        if not args.use_persistent_ckpt_worker:
            warn_rank_0(
                '--async-save is not supported without --use-persistent-ckpt-worker. '
                'Disabling --async-save.'
            )
            args.async_save = False

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

    if args.cuda_graph_impl == "local" and args.expert_model_parallel_size > 1:
       assert args.moe_pad_experts_for_cuda_graph_inference, \
        "--moe-pad-experts-for-cuda-graph-inference must be set when using CUDA graphs with expert parallelism"

    # MoE upcycling check
    if args.moe_use_upcycling:
        assert args.save is not None, "When using upcycling, the --save option must be specified."
        if not args.no_load_optim:
            args.no_load_optim = True
            warn_rank_0('enabling --no-load-optim for upcycling.')
        if not args.no_load_rng:
            args.no_load_rng = True
            warn_rank_0('enabling --no-load-rng for upcycling.')

    # --skip-train checks.
    if args.skip_train and not args.no_load_optim:
        args.no_load_optim = True
        warn_rank_0('enabling --no-load-optim when skipping training.')

    # Muon optimizer check
    if 'muon' in args.optimizer:

        # TODO: remove these checks once we support them
        assert not args.overlap_grad_reduce, "Muon optimizer does not support overlap grad reduce for now."
        assert not args.overlap_param_gather, "Muon optimizer does not support overlap param gather for now."

        assert not args.use_distributed_optimizer, "Muon optimizer does not support distributed optimizer for now."
        assert not args.use_torch_fsdp2, "Muon optimizer does not support Torch-FSDP2 for now."
        assert not args.use_megatron_fsdp, "Muon optimizer does not support Megatron-FSDP for now."
        assert args.ckpt_format in ["torch", "torch_dist"], "Muon optimizer supports torch and torch_dist checkpoint format."

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
        warn_rank_0("--replication-jump was specified despite not using replication. Ignoring.")
        args.replication_jump = None

    if args.delay_wgrad_compute:
        assert args.transformer_impl == 'transformer_engine', \
            "Delaying wgrad compute is only supported with transformer_engine implementation"
        if args.overlap_grad_reduce:
            assert is_te_min_version("2.8.0"), (
                "overlap_grad_reduce is only supported with TE >= 2.8.0 when enabling delay_wgrad_compute"
            )
            wgrad_in_graph_scope = CudaGraphScope.attn in args.cuda_graph_scope or (
                CudaGraphScope.moe_router in args.cuda_graph_scope
                and args.moe_shared_expert_intermediate_size is not None
                and not args.moe_shared_expert_overlap
            )
            if wgrad_in_graph_scope:
                assert is_te_min_version(
                    "2.12.0"
                ), "CUDA graph with delay_wgrad_compute requires TE version >= 2.12.0."
                assert args.gradient_accumulation_fusion, (
                    'CUDA graph with delay_wgrad_compute requires gradient_accumulation_fusion '
                    'to be enabled. This is because the default gradient accumulation does not '
                    'use static memory addresses, which breaks CUDA graph requirements.'
                )
                if CudaGraphScope.attn in args.cuda_graph_scope:
                    assert (
                        not args.add_bias_linear and not args.add_qkv_bias
                    ), "CUDA graph with delay_wgrad_compute doesn't support attn bias for now."

        if not args.gradient_accumulation_fusion:
            assert is_te_min_version("2.7.0"), (
                "disabling gradient_accumulation_fusion is only supported with TE >= 2.7.0 "
                "when enabling delay_wgrad_compute"
            )

    if args.fine_grained_activation_offloading:
        assert args.transformer_impl == 'transformer_engine', \
            "Fine-grained activation offloading is only supported with transformer_engine implementation"
        if is_te_min_version("2.10.0"):
            assert os.getenv("NVTE_CPU_OFFLOAD_V1", "0") == "1", \
                "For fine-grained activation offloading with TE >= 2.10.0, NVTE_CPU_OFFLOAD_V1 should be set to 1 to avoid offloading weights."

    if args.mtp_num_layers:
        assert not args.use_legacy_models, "The legacy Megatron models does not support Multi-Token Prediction (MTP)."
        # MTP is compatible with position embedding types that use position_ids.
        supported_position_types = ["learned_absolute", "rope", "mrope", "none"]
        assert args.position_embedding_type in supported_position_types, (
            f"Multi-Token Prediction (MTP) is not supported with '{args.position_embedding_type}' position embedding type. "
            f"The supported position embedding types are: {', '.join(supported_position_types)}."
        )

    if args.cpu_offloading_num_layers > 0:
        args.cpu_offloading = True

    # CUDA Graphs
    if args.cuda_graph_impl != "none":
        if (
            "transformer_engine" in (args.transformer_impl, args.cuda_graph_impl)
            and not args.te_rng_tracker
        ):
            args.te_rng_tracker = True
            warn_rank_0("te_rng_tracker is not enabled, enabling it for CUDA graphs.", args.rank)
        if args.cuda_graph_impl == "transformer_engine":
            assert (
                "expandable_segments:True" not in os.getenv("PYTORCH_CUDA_ALLOC_CONF", "")
                or os.getenv("NCCL_GRAPH_REGISTER", "") == "0"
            ), (
                "Setting NCCL_GRAPH_REGISTER=0 to avoid illegal memory access when using "
                "CUDA Graph with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True."
            )
    if args.cuda_graph_scope == "full" or (
        isinstance(args.cuda_graph_scope, list) and "full" in args.cuda_graph_scope
    ):
        if isinstance(args.cuda_graph_scope, list):
            assert args.cuda_graph_scope == ["full"], "full scope cannot be used with other scopes."
        args.cuda_graph_scope = []
        warn_rank_0(
            'full scope is deprecated. Use empty cuda_graph_scope to capture the whole layer.'
        )
    
    if args.multi_latent_attention:
        assert not args.group_query_attention, "Group query attention is mutually exclusive with multi latent attention."

    # MoE latent projections
    if args.moe_latent_size is not None:
        assert args.moe_latent_size > 0, "MoE latent projection dimension has to be greater than zero."
        assert args.num_experts is not None, "MoE latent projections are applicable only for MoE models."
        assert not args.use_legacy_models, "MoE latent projections are only supported for mcore models."
        assert not args.moe_use_legacy_grouped_gemm, "MoE latent projection is not supported yet with legacy grouped GEMM."

    if args.tiktoken_special_tokens and not args.tokenizer_special_tokens:
        warn_rank_0(
            "--tiktoken-special-tokens argument is deprecated and will be removed soon. "
            "Use --tokenizer-special-tokens instead."
        )
        args.tokenizer_special_tokens = args.tiktoken_special_tokens

    # Print arguments.
    _print_args("arguments", args)

    return args


def _print_args(title, args):
    """Print arguments."""
    from megatron.training.utils import is_rank0
    if is_rank0():
        print(f'------------------------ {title} ------------------------', flush=True)
        str_list = []
        for arg in vars(args):
            dots = '.' * (48 - len(arg))
            str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))
        for arg in sorted(str_list, key=lambda x: x.lower()):
            print(arg, flush=True)
        print(f'-------------------- end of {title} ---------------------', flush=True)


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

    kw_args['moe_latent_size'] = args.moe_latent_size

    if args.te_precision_config_file:
        assert not 'quant_recipe' in kw_args, "Quantization recipe already configured."
        # TODO(kwyss): Prohibit fp8_params or fp4_params with this flexibility
        kw_args['quant_recipe'] = load_quantization_recipe(args.te_precision_config_file)

    if hasattr(args, "use_kitchen_attention"):
        kw_args['use_kitchen_attention'] = args.use_kitchen_attention
    if hasattr(args, "kitchen_attention_backend"):
        kw_args['kitchen_attention_backend'] = args.kitchen_attention_backend

    # Return config.
    return config_class(**kw_args)


def _add_transformer_engine_args(parser):
    group = parser.add_argument_group(title='Transformer-Engine')

    # delayed scaling only configs
    group.add_argument('--fp8-param-gather', action='store_true',
                       help='Keep the compute param in fp8 (do not use any other intermediate '
                            'dtype) and perform the param all-gather in fp8.')

    # FP4 related arguments
    group.add_argument('--te-precision-config-file', default=None,
                       help='Configuration file to select per-module precision overrides. '
                       'See TransformerEngineMixedPrecision.md')
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
    group.add_argument('--cuda-graph-scope', nargs='+', type=lambda scope: CudaGraphScope[scope] if scope != "full" else scope, default=[],
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
                       'If not specified, the default scope is to capture the whole Transformer layer. '
                       'For backward compatibility, we still allow passing "full" to specify capturing the whole layer, and convert it to an empty list.')
    group.add_argument('--use-legacy-static-engine', action='store_true', default=False,
                       help='Use legacy static engine. (Current static engine uses dynamic engine under the hood)',
                       dest='use_legacy_static_engine')
    group.add_argument('--inference-max-requests', type=int, default=8,
                       help='Maximum number of requests for inference.',
                       dest='inference_max_requests')
    group.add_argument('--inference-max-seq-length', type=int, default=2560,
                       help='Maximum sequence length expected for inference (prefill + decode).',
                       dest='inference_max_seq_length')
    group.add_argument('--inference-dynamic-batching',
                       action='store_true', default=False,
                       help='Enable dynamic batching mode.')
    group.add_argument('--inference-dynamic-batching-buffer-size-gb',
                       type=float, default=40.,
                       help='Amount of on-GPU memory allocated for the KV cache. '
                       'The total amount of memory allocated for the KV cache '
                       '(CPU + GPU memory) depends on the value set for the '
                       'unified virtual memory (UVM) level (via '
                       '`--inference-dynamic-batching-unified-memory-level`).'
                       'If the UVM level is 0, then only GPU memory is used and '
                       'the total memory equals `buffer_size_gb`. If the UVM '
                       'level is 1, then additional memory is utilized on the '
                       'CPU and the total memory equals `buffer_size_gb + '
                       'paused_buffer_size_gb`.')
    group.add_argument('--inference-dynamic-batching-paused-buffer-size-gb',
                       type=float, default=None,
                       help='Amount of memory reserved for paused requests in '
                       'the dynamic inference context. Active requests are '
                       'paused when there are not enough active blocks available '
                       'to continue generating a request.')
    group.add_argument('--inference-dynamic-batching-mamba-memory-ratio', type=float, default=None,
                       help='Percentage of memory buffer to allocate for Mamba states. '
                       'If not specified, allocates Mamba state tensors for each KV cache block. '
                       'Only used for hybrid models.')
    group.add_argument('--inference-dynamic-batching-block-size',
                       type=int, default=256,
                       help='KV cache block size. '
                       'It should be a multiple of 256')
    group.add_argument('--inference-dynamic-batching-max-requests',
                       type=int, default=None,
                       help='Override the inference context\'s `max_requests`. '
                       'By default, `max_requests` is set to the number of '
                       'blocks in the context\'s memory buffer.')
    group.add_argument('--inference-dynamic-batching-max-tokens',
                       type=int, default=None,
                       help='Override the inference context\'s default `max_tokens`.')
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
    group.add_argument('--inference-dynamic-batching-track-generated-token-events',
                       action='store_true',
                       help='Track per-token events with timestamps for each generated token. '
                       'When enabled, each generated token creates a GENERATED_TOKEN event '
                       'with a timestamp, useful for per-token latency analysis.')
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
    # TODO(ksanthanam): Clean this up in future PR
    group.add_argument('--enable-chunked-prefill', dest='enable_chunked_prefill',
                       action='store_true', default=False,
                       help="Enable chunked prefill (disabled by default)")
    group.add_argument('--inference-dynamic-batching-cuda-graph-max-tokens',
                       type=int, default=16384,
                       help='Maximum number of tokens to capture in a cuda graph.')
    group.add_argument('--inference-dynamic-batching-cuda-graph-mixed-prefill-count',
                       type=int, default=16,
                       help='Number of mixed prefill requests to capture in a cuda graph.')
    group.add_argument('--inference-logging-step-interval', type=int, default=0,
                       help='Step interval for logging inference metrics. '
                            'Default to 0 to disable inference logging.')
    group.add_argument('--inference-wandb-logging', action=argparse.BooleanOptionalAction,
                       required=False, default=False, help='Enable inference wandb logging.')
    group.add_argument("--inference-coordinator-port", type=int, default=12346,
                       help="This port will be used to setup the inference coordinator on node-0")
    return parser


def _add_network_size_args(parser):
    exclude = [
        # cannot provide callables over CLI
        "timers",
        "finalize_model_grads_func",
        "grad_scale_func",
        "no_sync_func",
        "grad_sync_func",
        "param_sync_func",
        "_cpu_offloading_context",
        "init_method",
        "output_layer_init_method",
        "embedding_init_method",
        "activation_func",
        # types affect docstring
        "pipeline_model_parallel_layout",
        "window_size",
        "window_attn_skip_freq",
        "no_rope_freq",
        "moe_layer_freq",
        "linear_attention_freq",
        "moe_router_load_balancing_type",
        "moe_aux_loss_coeff",
        "cp_comm_type",
        "cuda_graph_scope",
        # no CLI argument exists for these
        "virtual_pipeline_model_parallel_size",
        "params_dtype",
        "enable_autocast",
        "autocast_dtype",
        "num_microbatches_with_partial_activation_checkpoints",
        "tp_comm_overlap_disable_qkv",
        "tp_comm_overlap_disable_fc1",
        "pipeline_dtype",
        "variable_seq_lengths",
        "batch_p2p_comm",
        "batch_p2p_sync",
        "deallocate_pipeline_outputs",
        "cpu_offloading",
        "cpu_offloading_activations",
        "cpu_offloading_weights",
        "cpu_offloading_double_buffering",
        "num_layers_in_first_pipeline_stage",
        "num_layers_in_last_pipeline_stage",
        "softmax_scale",
        "gated_linear_unit",
        "bias_activation_fusion",
        "activation_func_fp8_input_store",
        "test_mode",
        "memory_efficient_layer_norm",
        "fused_single_qkv_rope",
        "fp8_dot_product_attention",
        "fp8_multi_head_attention",
        "tp_only_amax_red",
        "use_kitchen",
        "moe_token_dropping",
        "cuda_graph_use_single_mempool",
        "cuda_graph_retain_backward_graph",
        "disable_parameter_transpose_cache",
        "inference_sampling_seed",
        "use_inference_optimized_layers",
        "heterogeneous_block_specs",
        "hetereogenous_dist_checkpoint",
        "quant_recipe",
        # deprecated and no CLI arg exists
        "tp_comm_atomic_ag",
        "tp_comm_atomic_rs",
        "moe_router_topk_limited_devices",
        # already generated by another config
        "inference_rng_tracker",
        "use_te_rng_tracker",
        "log_max_attention_logit",
        "barrier_with_L1_time",
        # args uses same var with a different name
        "num_moe_experts",
        "fp8_param",
        # incompatible defaults in dataclass
        "gradient_accumulation_fusion",
        "overlap_p2p_comm",
        "attention_softmax_in_fp32",
        "masked_softmax_fusion",
        "persist_layer_norm",
        "bias_dropout_fusion",
        "apply_rope_fusion",
    ]
    transformer_factory = ArgumentGroupFactory(TransformerConfig, exclude=exclude)
    transformer_group = transformer_factory.build_group(parser, "transformer configuration")

    group = parser.add_argument_group(title='network size')

    group.add_argument('--encoder-num-layers', type=int, default=None,
                       help='Number of encoder transformer layers.')
    group.add_argument('--decoder-num-layers', type=int, default=None,
                       help='Number of decoder transformer layers.')
    group.add_argument('--group-query-attention', action='store_true',
                          help='Use group-query attention.')
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
    group.add_argument('--make-vocab-size-divisible-by', type=int, default=128,
                       help='Pad the vocab size to be divisible by this value.'
                       'This is added for computational efficieny reasons.')
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
    group.add_argument('--onnx-safe', type=bool, required=False,
                       help='Use workarounds for known problems with '
                       'Torch ONNX exporter')
    group.add_argument('--bert-no-binary-head', action='store_false',
                       help='Disable BERT binary head.',
                       dest='bert_binary_head')
    group.add_argument('--untie-embeddings-and-output-weights', action='store_true',
                       help='Untie embeddings and output weights.')
    return parser

def _add_straggler_detector_args(parser):
    from megatron.training.resilience_config import StragglerDetectionConfig

    straggler_factory = ArgumentGroupFactory(StragglerDetectionConfig)
    group = straggler_factory.build_group(parser, "straggler")

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
    group.add_argument('--ft-num-warmup-iters', type=int, default=5,
                       help='Number of warmup iterations before monitoring step section and '
                       'out-of-section timeouts. The first N iterations are excluded from '
                       'timeout monitoring as they can be significantly slower than steady-state. '
                       'Default: 5. Note: This feature is for Nvidia internal use only.')
    return parser


def _add_logging_args(parser):
    from megatron.training.training_config import LoggerConfig

    log_factory = ArgumentGroupFactory(LoggerConfig, exclude = ["log_throughput_to_tensorboard", "throughput_window_size", "memory_keys", "log_l2_norm_grad_to_tensorboard", "log_runtime_to_tensorboard", "runtime_time_unit", "filter_warnings", "modules_to_filter", "set_level_for_all_loggers", "save_config_filepath"])
    group = log_factory.build_group(parser, title="logging")

    return parser


def _add_regularization_args(parser):
    group = parser.add_argument_group(title='regularization')

    group.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay coefficient for L2 regularization.')
    group.add_argument('--apply-wd-to-qk-layernorm', action='store_true',
                       help='Apply weight decay to qk layernorm as a special case.')
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
    group.add_argument('--muon-momentum', type=float, default=0.9,
                       help='Momentum factor for Muon optimizer')
    group.add_argument('--muon-no-split-qkv', action='store_false', default=True,
                       dest='muon_split_qkv',
                       help='Whether to split QKV parameters for Muon optimizer')
    group.add_argument('--muon-use-nesterov', action='store_true',
                       help='Whether to use Nesterov-style momentum in the internal SGD')
    group.add_argument('--muon-scale-mode', type=str, default='spectral',
                       choices=['spectral', 'unit_rms_norm', 'shape_scaling'],
                       help='Scale mode for Muon optimizer')
    group.add_argument('--muon-fp32-matmul-prec', type=str, default='medium',
                       choices=['low', 'medium', 'high'],
                       help='FP32 matmul precision for Newton-Schulz iteration')
    group.add_argument('--muon-num-ns-steps', type=int, default=5,
                       help='Number of Newton-Schulz steps for Muon optimizer')
    group.add_argument('--muon-tp-mode', type=str, default='blockwise',
                       choices=['blockwise', 'duplicated', 'distributed'],
                       help='How to perform NS calculation for tensor model parallel weights')
    group.add_argument('--muon-extra-scale-factor', type=float, default=1.0,
                       help='Additional scale factor for the muon update')

    group.add_argument('--no-weight-decay-cond-type', type=str, choices=['apply_wd_to_qk_layernorm'],
                       help='Type of no weight decay condition. Choices: '
                       'None (default): apply weight decay to 1D weights and biases.'
                       '"apply_wd_to_qk_layernorm": additionally apply weight decay to '
                       'qk layernorm as a special case.'
                       'DEPRECATED. Please use --apply-wd-to-qk-layernorm instead. ')
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
    group.add_argument('--langrl-env-config', type=str, default=None,
                       help="Path to YAML config file for RL environment configuration.")
    group.add_argument('--rl-default-temperature', type=float, default=1.0,
                       help="Default temperature for model inference.")
    group.add_argument('--rl-default-top-p', type=float, default=0,
                       help="Default top-p for model inference.")
    group.add_argument('--rl-default-top-k', type=int, default=-1,
                       help="Default top-k for model inference.")
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
    group.add_argument('--rl-use-sequence-packing', action=argparse.BooleanOptionalAction, type=bool, default=False,
                       help='Enable sequence packing')
    group.add_argument('--rl-sequence-packing-max-sequences-per-bin', type=int, default=50,
                       help='Maximum number of sequences that can be packed into a single bin. ')
    group.add_argument('--rl-sequence-packing-algo', type=str, default='fifo',
                       choices=['fifo', 'round-robin'],
                       help='Algorithm for distributing packed bins across ranks. '
                            'fifo: first-in-first-out sequential distribution, '
                            'round-robin: distribute bins cyclically across ranks for better load balancing')
    group.add_argument('--rl-training-cuda-graphs', action=argparse.BooleanOptionalAction, type=bool,
                       default=False,
                       help='If set, do not call `delete_cuda_graphs` or `toggle_cuda_graphs` when the inference engine is suspended.')
    group.add_argument('--rl-inference-tensor-model-parallel-size', type=int, default=None,
                       help='Degree of tensor model parallelism for inference for RL.')     
    group.add_argument(
        '--rl-inference-pipeline-model-parallel-size',
        type=int,
        default=None,
        help='Degree of pipeline model parallelism for inference for RL.',
    )
    group.add_argument(
        '--rl-inference-expert-model-parallel-size',
        type=int,
        default=None,
        help='Degree of expert model parallelism for inference for RL.',
    )
    group.add_argument(
        '--rl-inference-expert-tensor-model-parallel-size',
        type=int,
        default=None,
        help='Degree of expert tensor model parallelism for inference for RL. '
             'For MoE models, this controls the TP size for expert layers specifically. '
             'Defaults to training expert_tensor_parallel_size if not specified.',
    )
    group.add_argument(
        '--rl-inference-model-unified-memory-level',
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            'Allocate the separate RL inference model parameters from a unified virtual memory (UVM) '
            'CUDA mempool. Level 0 disables UVM (default). Level 1 enables UVM allocation so the '
            'inference model weights can be prefetched to CPU when idle while keeping CUDA-graph-safe '
            'device pointers.'
        ),
    )
    group.add_argument(
        '--rl-offload-inference-model-weights-when-idle',
        action=argparse.BooleanOptionalAction,
        required=False,
        default=False,
        help=(
            'When using a separate RL inference model, offload its weights to CPU when not doing rollout '
            'inference, and restore to GPU right before inference. Works with two backends: '
            '1) UVM (when --rl-inference-model-unified-memory-level=1), or '
            '2) torch_memory_saver (when UVM is not enabled; requires torch_memory_saver to be installed).'
        ),
    )
    group.add_argument('--refit-method', type=str, default='gloo',
                       choices=['nccl', 'gloo', 'nvshmem'],
                       help=('Method to refit the model weights between training and inference models during RL. '
                             'nccl: use NCCLCopyService to refit using NCCL; '
                             'gloo: use GlooCopyService over CPU; '
                             'nvshmem: use NVSHMEMCopyService to refit using the NVSHMEM.'))
    group.add_argument('--rl-verify-model-weights-swap', action=argparse.BooleanOptionalAction, default=False,
                       help='If set, verify that the model weights were correctly transferred by comparing forward pass outputs on'
                       'the first swap of model weights.')

    group.add_argument('--rl-parallel-generation-tasks', type=int, default=512,
                        help='Number of parallel generation tasks for RL inference.')
    group.add_argument('--rl-skip-bos-token', action=argparse.BooleanOptionalAction, type=bool, default=False,
                        help='Skip BOS token at the beginning of the sequences. Default is False.')
    return parser

def _add_training_args(parser):
    from megatron.training.training_config import TrainingConfig
    from megatron.training.common_config import ProfilingConfig

    prof_factory = ArgumentGroupFactory(ProfilingConfig, exclude=["record_shapes", "nvtx_ranges"])
    prof_group = prof_factory.build_group(parser, "profiling")

    train_factory = ArgumentGroupFactory(TrainingConfig)
    group = train_factory.build_group(parser, "training")

    group.add_argument('--batch-size', type=int, default=None,
                       help='Old batch size parameter, do not use. '
                       'Use --micro-batch-size instead')
    group.add_argument('--recompute-activations', action='store_true',
                       help='recompute activation to allow for training '
                       'with larger models, sequences, and batch sizes.')
    group.add_argument('--no-check-for-nan-in-loss-and-grad', action='store_false',
                       help='Check for NaNs in loss and grad',
                       dest='check_for_nan_in_loss_and_grad')
    group.add_argument('--check-for-large-grads', action='store_true',
                       help='Check for unexpectedly large grads',
                       dest='check_for_large_grads')
    group.add_argument('--result-rejected-tracker-filename', type=str, default=None,
                       help='Optional name of file tracking `result_rejected` events.')
    group.add_argument('--tp-comm-overlap-cfg', type=str, default=None,
                       help='Config file when tp_comm_overlap is enabled.')

    # deprecated
    group.add_argument('--checkpoint-activations', action='store_true',
                       help='Checkpoint activation to allow for training '
                       'with larger models, sequences, and batch sizes.')
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
    group.add_argument('--use-flash-attn', action='store_true',
                       help='use FlashAttention implementation of attention. '
                       'https://arxiv.org/abs/2205.14135')
    group.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'sgd', 'muon', 'dist_muon'],
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
    group.add_argument('--no-persist-layer-norm', action='store_true',
                       help='Disable using persistent fused layer norm kernel. '
                       'This kernel supports only a set of hidden sizes. Please '
                       'check persist_ln_hidden_sizes if your hidden '
                       'size is supported.')
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

    return parser


def _add_rerun_machine_args(parser):
    from megatron.training.resilience_config import RerunStateMachineConfig

    rerun_factory = ArgumentGroupFactory(RerunStateMachineConfig, exclude=["check_for_nan_in_loss"])
    group = rerun_factory.build_group(parser, "rerun engine")

    return parser


def _add_initialization_args(parser):
    from megatron.training.common_config import RNGConfig

    rng_factory = ArgumentGroupFactory(RNGConfig)
    group = rng_factory.build_group(parser, "RNG and initialization")

    group.add_argument('--init-method-xavier-uniform', action='store_true',
                       help='Enable Xavier uniform parameter initialization')

    return parser


def _add_learning_rate_args(parser):
    from megatron.training.training_config import SchedulerConfig

    sched_factory = ArgumentGroupFactory(SchedulerConfig, exclude=["no_weight_decay_cond_type"])
    group = sched_factory.build_group(parser, title="learning rate and weight decay")

    group.add_argument('--lr', type=float, default=None,
                       help='Initial learning rate. Depending on decay style '
                       'and initial warmup, the learning rate at each '
                       'iteration would be different.')
    group.add_argument('--warmup', type=int, default=None,
                       help='Old lr warmup argument, do not use. Use one of the'
                       '--lr-warmup-* arguments above')
    group.add_argument('--min-lr', type=float, default=0.0,
                       help='Minimum value for learning rate. The scheduler'
                       'clip values below this threshold.')
    group.add_argument('--decoupled-lr', type=float, default=None,
                       help='Separate learning rate for the input and output layer')
    group.add_argument('--decoupled-min-lr', type=float, default=None,
                       help='Minimum value for learning rate for the input and output layer. The scheduler'
                       'clip values below this threshold')

    return parser


def _add_checkpointing_args(parser):
    from megatron.training.training_config import CheckpointConfig

    ckpt_factory = ArgumentGroupFactory(CheckpointConfig, exclude=["most_recent_k", "save_tokenizer_assets", "save_optim", "save_rng", "load_optim", "load_rng"])
    group = ckpt_factory.build_group(parser, "checkpointing")

    group.add_argument('--no-save-optim', action='store_true', default=None,
                       help='Do not save current optimizer.')
    group.add_argument('--no-save-rng', action='store_true', default=None,
                       help='Do not save current rng state.')
    group.add_argument('--no-load-optim', action='store_true', default=None,
                       help='Do not load optimizer when loading checkpoint.')
    group.add_argument('--no-load-rng', action='store_true', default=None,
                       help='Do not load rng state when loading checkpoint.')
    group.add_argument('--use-dist-ckpt', action='store_true',
                       dest='use_dist_ckpt_deprecated',
                       help='Deprecated: see --ckpt-format.')
    group.add_argument('--dist-ckpt-format',
                       dest='dist_ckpt_format_deprecated',
                       help='Deprecated: see --ckpt-format.')
    group.add_argument('--ckpt-fully-parallel-save', action='store_true',
                       dest='ckpt_fully_parallel_save_deprecated',
                       help='Deprecated: see --no-ckpt-fully-parallel-save.')
    return parser


def _add_mixed_precision_args(parser):
    group = parser.add_argument_group(title='mixed precision')

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
    group.add_argument('--attention-softmax-in-fp32', action='store_true',
                       help='Run attention masking and softmax in fp32.')
    group.add_argument('--accumulate-allreduce-grads-in-fp32',
                       action='store_true',
                       help='Gradient accumulation and all-reduce in fp32.')
    group.add_argument('--fp16-lm-cross-entropy', action='store_true',
                       help='Move the cross entropy unreduced loss calculation'
                       'for lm head to fp16.')
    group.add_argument('--reuse-grad-buf-for-mxfp8-param-ag', action='store_true',
                       help='If True, reuse the grad buffer for MXFP8 parameter all-gather.')

    return parser


def _add_distributed_args(parser):
    from megatron.training.common_config import DistributedInitConfig

    dist_init_factory = ArgumentGroupFactory(DistributedInitConfig)
    group = dist_init_factory.build_group(parser, "distributed init")

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
    group.add_argument('--no-overlap-p2p-communication', action='store_false',
                       help='overlap pipeline parallel communication with forward and backward chunks in 1F1B',
                       dest='overlap_p2p_comm')
    group.add_argument('--overlap-grad-reduce', action='store_true',
                       default=False, help='If set, overlap DDP grad reduce.')
    group.add_argument('--ddp-num-buckets', type=int, default=None,
                       help='Number of buckets for data-parallel communication')
    group.add_argument('--ddp-bucket-size', type=int, default=None,
                       help='Bucket size for data-parallel communication')
    group.add_argument('--ddp-pad-buckets-for-high-nccl-busbw', action='store_true',
                       default=False, help='If set, make sure the bucket size is divisible by a large power '
                       'of 2 (2^16) to ensure NCCL collectives have high bus bandwidth at large DP counts, '
                       'since NCCL message size (which for ring algorithms is bucket_size / dp_size) '
                       'apparently needs to be divisible by a power of 2 for high busbw.')
    group.add_argument('--ddp-reduce-scatter-with-fp32-accumulation', action='store_true',
                       default=False, help='If set, use a reduce-scatter implementation which sends lower-precision '
                       'values over the wire (using an all-to-all to keep total communication overhead in line '
                       'with the standard ring implementation) but performs accumulation locally in FP32.')
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
    group.add_argument('--use-distributed-optimizer', action='store_true',
                       help='Use distributed optimizer.')
    group.add_argument('--use-nccl-ub', action='store_true', dest='nccl_ub',
                       help='Use the userbuffer registration for DP/FSDP communication buffers.'
                       'This option will reduce GPU SM usage for the DP/FSDP communication,'
                       'which is improving the performance of the overlapped computation.')
    group.add_argument('--disable-symmetric-registration', action='store_true', dest='disable_symmetric_registration',
                       default=False, help='Disable symmetric (window) registration for NCCL userbuffer registration.'
                       'This option will force to use conventional (local) userbuffer registration when use-nccl-ub is set.')
    group.add_argument('--fsdp-manual-registration', action='store_true', dest='fsdp_manual_registration',
                       default=False, help='Manually register the FSDP communication buffers to NCCL user buffer.'
                       'This option is only effective when use-megatron-fsdp and use-nccl-ub is set.')
    group.add_argument('--create-all-gather-group', action='store_true',
                   help='Create a separate process group for all-gather operations '
                   'to overlap reduce-scatter and all-gather operations.')
    group.add_argument('--data-parallel-sharding-strategy', type=str, default='no_shard',
                       choices=['no_shard', 'optim', 'optim_grads', 'optim_grads_params'],
                       help='Sharding strategy of data parallelism.')
    group.add_argument('--outer-dp-sharding-strategy', type=str, default='no_shard',
                       choices=['no_shard', 'optim'],
                       help='Sharding strategy for outer data parallel group in Hybrid Sharded Data Parallel (HSDP) mode. '
                            'Valid values are "no_shard" (DP Replication) and "optim" (Optimizer State Hybrid Sharding). '
                            'The "optim" option is only supported when --data-parallel-sharding-strategy is "optim_grads_params". '
                            'This option is only effective when Hybrid FSDP is enabled (i.e., when dp_outer_dim is not None). '
                            'Default: "no_shard".')
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
    group.add_argument('--torch-fsdp2-no-reshard-after-forward', action='store_false', dest='torch_fsdp2_reshard_after_forward',
                       help='Whether to reshard weights after forward pass when using PyTorch FSDP2. '
                       'Set to enable FSDP ZeRO-2.')
    group.add_argument('--cp-comm-type', nargs='+', type=str, default=["p2p"],
                       help='Inter-gpu communication type for context parallelism: '
                       'p2p, a2a, allgather or a2a+p2p. If a single string is provided, '
                       'all layers will share the same communication type. Users can also '
                       'specify separated types for each layer like '
                       '--cp-comm-type p2p p2p a2a a2a a2a+p2p a2a+p2p')
    group.add_argument('--fake-process-group', action='store_true', default=False,
                       help='If set, initialize with fake distributed process group and all distributed communication operations will be skipped. \
                       This is quite useful for profiling memory usage of distributed training with just one GPU. \
                       Setting WORLD_SIZE and RANK to the specific values for target distribtued scale.')
    return parser


def _add_validation_args(parser):
    from megatron.training.training_config import ValidationConfig

    val_factory = ArgumentGroupFactory(ValidationConfig)
    group = val_factory.build_group(parser, "validation")

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
    group.add_argument('--tokenizer-special-tokens', type=str, nargs='+', default=None,
                       help='List of special tokens. For TikTokenizer needs to have '
                            '["<unk>", "<s>", "</s>", "<mask>", "<pad>", "<cls>", "<sep>"]')
    group.add_argument('--tiktoken-pattern', type=str, default=None,
                       help='Which tiktoken pattern to use. Options: [v1, v2]')
    group.add_argument('--tiktoken-num-special-tokens', type=int, default=1000,
                       help='Number of special tokens in tiktoken tokenizer')
    group.add_argument('--tiktoken-special-tokens', type=str, nargs='+', default=None,
                       help='List of tiktoken special tokens, needs to have '
                            '["<unk>", "<s>", "</s>", "<mask>", "<pad>", "<cls>", "<sep>"]')
    group.add_argument('--tokenizer-sentencepiece-legacy', action='store_true', default=False,
                       help='SentencePiece tokenizer wrapper legacy behavior. Allows special tokens usage.')
    group.add_argument('--tokenizer-hf-use-fast', action='store_true', default=False,
                       help='Whether to use fast HuggingFace tokenizer.')
    group.add_argument('--tokenizer-hf-include-special-tokens', action='store_true', default=False,
                       help='Converting text to ids will include special for HuggingFace tokenizer.')
    group.add_argument("--trust-remote-code", action="store_true", default=False,
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
    group.add_argument('--phase-transition-iterations', type=str, default=None,
                       help='Comma-separated list of iterations where phase '
                       'transitions occur. Requires fixed global batch size across phases. '
                       'Does not support batch size ramp-up.')
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
    group.add_argument('--per-dataset-sequences-path', default=None,
                       help='Path to a json file with the sequences per dataset. Check the tools/build_sequences_per_dataset.py script to build this file.')
    group.add_argument('--dataloader-fast-cache-load', action='store_true',
                       help='Option to use the fast cache loading path when building the datasets. Requires all the dataset caches to be built and stored in --data-cache-path.')
    group.add_argument('--dataloader-defer-npy-index-mmap', action='store_true',
                       help='Defer the mmap of the dataset indexes (.npy files) until the first access. Requires all the dataset caches to be built and stored in --data-cache-path.')
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
    group.add_argument('--allow-ambiguous-pad-tokens', action='store_true',
                       help='Whether to prevent pad tokens already present in the dataset '
                       'from being masked out when the pad token incorrectly shares the same id '
                       'with other special tokens in the tokenizer. Note that this argument has '
                       'no effect when the tokenizer correctly provides a unique id for the pad. '
                       'Masking out such ambiguous pad tokens results in training instability. '
                       'Such a scenario is best resolved by fixing the tokenizer; leaving this '
                       'option as False provides a workaround. '
                       'When left to the default of False, any token ids that collide with the '
                       'pad token id - as provided by the tokenizer - will not be masked out of '
                       'the loss calculation: it cannot be determined whether they are truly pad. '
                       'If instead this argument is set, the training flow will treat all tokens '
                       'that share the same id as the pad token as true pad tokens, potentially '
                       'causing severe training instability.')
    group.add_argument('--fim-data', action='store_true', help='Whether to use the FIM dataset.')
    group.add_argument('--fim-rate', type=float, default=0.5,
                       help='Probability to convert a training sample into a FIM format.')
    group.add_argument('--fim-spm-rate', type=float, default=0.5,
                       help='Probability that the a FIM sample uses the SPM format over the PSM format.')
    group.add_argument('--fim-split-sample', type=str, default=None,
                       help='String around which to split the sample for FIM.')
    group.add_argument('--fim-fragment-rate', type=float, default=None,
                       help='Rate of FIM on each fragment when --fim-split-sample is not None.')
    group.add_argument('--fim-no-prefix', type=str, default=None,
                       help='Do not apply FIM to fragments that start with this prefix')
    group.add_argument('--fim-prefix-token', type=str, default='<fim_prefix>',
                       help='FIM prefix token')
    group.add_argument('--fim-middle-token', type=str, default='<fim_middle>',
                       help='FIM middle token')
    group.add_argument('--fim-suffix-token', type=str, default='<fim_suffix>',
                       help='FIM suffix token')
    group.add_argument('--fim-pad-token', type=str, default='<fim_pad>',
                       help='FIM PAD token')
    group.add_argument('--fim-eod-token', type=str, default='<|endoftext|>',
                       help='FIM EOD token')
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

def _add_moe_args(parser):
    group = parser.add_argument_group(title="moe")
    # General arguments
    group.add_argument('--num-experts', type=int, default=None,
                       help='Number of Experts in MoE (None means no MoE)')
    group.add_argument('--moe-layer-freq', type=moe_freq_type, default=1,
                       help='Frequency between MoE layers and Dense layers. Accepts either: '
                            '- An integer N: Represents a 1:N ratio, meaning one expert layer for every N-1 dense layers '
                            '- A string containing a Python list expression that defines a custom pattern, e.g.: '
                            '"([1]*3+[0]*1)*3" evaluates to [1,1,1,0,1,1,1,0,1,1,1,0] '
                            'where 1 indicates an expert layer and 0 indicates a dense layer. '
                            'Examples: "([0]+[1]*23)": 1 dense layer followed by 23 expert layers, '
                            '"([1]*3+[0]*2)*2": Three expert layers followed by two dense layers, repeated twice.')
    group.add_argument('--moe-use-upcycling', action='store_true',
                       help='Load a checkpoint of a dense model, convert it into an MoE model, and save the converted model to the path specified by --save. '
                       'Upcycling is implemented on the top of distributed checkpointing, so it supports parallel modes different from the dense model.')
    # Router arguments
    group.add_argument('--moe-router-load-balancing-type', nargs='+', type=str,
                       choices=['aux_loss', 'seq_aux_loss', 'global_aux_loss', 'sinkhorn', 'none'],
                       default='aux_loss',
                       help='Determines the load balancing strategy for the router. "aux_loss" corresponds to the load balancing loss used in GShard and SwitchTransformer; "seq_aux_loss" corresponds to the load balancing loss used in DeepSeekV2, which computes the loss for each individual sample; "sinkhorn" corresponds to the balancing algorithm used in S-BASE, and "none" implies no load balancing. The default is "aux_loss".')
    group.add_argument('--moe-aux-loss-coeff', type=float, nargs='+', default=0.0,
                       help='Scaling coefficient for the aux loss: a starting value of 1e-2 is recommended.')
    # Token dispatcher arguments
    # MoE communication overlap arguments

    group.add_argument('--moe-upcycling-granularity', type=int, default=1,
                       help='This param sepecifics how many times smaller is the expert hidden size compared with the original dense FFN hidden size. '
                       'For using granular upcycling strategy, please set this param as a positive integer. If this param is set to 1, it means using the default upcycling strategy.')
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

def _add_experimental_attention_variant_args(parser):
    group = parser.add_argument_group(title="experimental_attention_variant")
    # Linear attention
    group.add_argument('--linear-attention-freq', type=la_freq_type, default=None,
                       help='Frequency between LA (linear attention) layers and'
                            ' SDPA (scaled dot-product attention) layers. Accepts either: '
                            '- An integer N: Represents a (N-1):N ratio, meaning (N-1) LA layers for every 1 SDPA layer '
                            '- A string containing a Python list expression that defines a custom pattern, e.g.: '
                            '"([1]*3+[0]*1)*3" evaluates to [1,1,1,0,1,1,1,0,1,1,1,0] '
                            'where 1 indicates an LA layer and 0 indicates a SDPA layer. '
                            'Examples: "([0]+[1]*23)": 1 SDPA layer followed by 23 LA layers, '
                            '"([1]*3+[0]*2)*2": Three LA layers followed by two SDPA layers, repeated twice.')
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
        from megatron.core.extensions.kitchen import KitchenSpecProvider, HAVE_KITCHEN

    except (ImportError, ModuleNotFoundError):
        HAVE_KITCHEN = False

    if HAVE_KITCHEN:
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
            help="Use a default kitchen recipe for all linear layers as defined by QAT_PARAMS index. "
            "The argument has no effect on attention layers.",
        )
    return parser

def _add_sft_args(parser):
    group = parser.add_argument_group(title='sft')
    group.add_argument('--sft', action="store_true", help='Megatron SFT training')
    group.add_argument('--sft-tokenizer-prompt-format', type=str, default="nemotron-h-aligned",
                       help='SFT prompt format.')
    return parser
