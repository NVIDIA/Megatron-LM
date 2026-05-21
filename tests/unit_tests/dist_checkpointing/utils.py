# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from functools import partial
from typing import Any, Callable, Tuple, Union
from unittest import mock

import torch

from megatron.core.dist_checkpointing.strategies.cached_metadata_filesystem_reader import (
    CachedMetadataFileSystemReader,
)
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.optimizer import ChainedOptimizer
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.training.arguments import parse_args
from megatron.training.training import get_model
from megatron.training.utils import unwrap_model

NUM_LAYERS = 8
HIDDEN_SIZE = 16
NUM_ATTENTION_HEADS = 8


def initialize_gpt_model(
    pre_process=True, post_process=True, seed=0, use_glu=True, **config_kwargs
):
    # These kwargs are passed through training.get_model for model construction,
    # but are not part of TransformerConfig; strip them before building config.
    config_kwargs.pop("pg_collection", None)
    config_kwargs.pop("config", None)

    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)

    default_config_kwargs = dict(
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_ATTENTION_HEADS,
        use_cpu_initialization=True,
        bf16=True,
    )
    default_config_kwargs.update(**config_kwargs)
    transformer_config = TransformerConfig(**default_config_kwargs, gated_linear_unit=use_glu)
    model = GPTModel(
        config=transformer_config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=128,
        max_sequence_length=4,
        pre_process=pre_process,
        post_process=post_process,
    )

    with torch.no_grad():
        for p in model.parameters():
            p.random_()
    return model


def initialize_moe_model(
    pre_process=True,
    post_process=True,
    seed=0,
    use_glu=True,
    use_sp=False,
    use_te=False,
    use_grouped_mlp=False,
    **config_kwargs,
):
    # These kwargs are passed through training.get_model for model construction,
    # but are not part of TransformerConfig; strip them before building config.
    config_kwargs.pop("pg_collection", None)
    config_kwargs.pop("config", None)

    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)
    expert_num = 8

    default_config_kwargs = dict(
        num_layers=8,
        hidden_size=16,
        num_attention_heads=8,
        use_cpu_initialization=True,
        num_moe_experts=expert_num,
        sequence_parallel=use_sp,
        moe_grouped_gemm=use_grouped_mlp,
        add_bias_linear=False,
    )
    default_config_kwargs.update(**config_kwargs)
    transformer_config = TransformerConfig(**default_config_kwargs, gated_linear_unit=use_glu)
    if use_te:
        spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=expert_num, moe_grouped_gemm=use_grouped_mlp
        )
    else:
        spec = get_gpt_layer_local_spec(num_experts=expert_num, moe_grouped_gemm=use_grouped_mlp)
    model = GPTModel(
        config=transformer_config,
        transformer_layer_spec=spec,
        vocab_size=128,
        max_sequence_length=4,
        pre_process=pre_process,
        post_process=post_process,
    )

    model.bfloat16()
    with torch.no_grad():
        for p in model.parameters():
            p.random_()
    return model


def init_basic_mock_args(args, tp, pp, bf16=True):
    args.data_parallel_random_init = False
    args.virtual_pipeline_model_parallel_size = None
    args.fp16 = False
    args.bf16 = bf16
    args.accumulate_allreduce_grads_in_fp32 = False
    args.overlap_grad_reduce = False
    args.overlap_param_gather_with_optimizer_step = False
    args.fp8_param_gather = False
    args.use_distributed_optimizer = True
    args.ddp_bucket_size = None
    args.check_for_nan_in_loss_and_grad = False
    args.ddp_average_in_collective = False
    args.tensor_model_parallel_size = tp
    args.pipeline_model_parallel_size = pp
    args.enable_ft_package = False
    args.use_torch_fsdp2 = False
    args.init_model_with_meta_device = False
    return args


def init_checkpointing_mock_args(args, ckpt_dir, fully_parallel=False):
    args.non_persistent_global_ckpt_dir = None
    args.non_persistent_ckpt_type = None
    args.save = ckpt_dir
    args.load = ckpt_dir
    args.pretrained_checkpoint = None
    args.ckpt_fully_parallel_save = fully_parallel
    args.ckpt_fully_parallel_load = fully_parallel
    args.async_save = False
    args.use_dist_ckpt = True
    args.ckpt_format = 'torch_dist'
    args.no_save_optim = False
    args.no_save_rng = False
    args.ckpt_assume_constant_structure = False
    args.ckpt_load_validate_sharding_integrity = True
    args.log_progress = False
    args.auto_detect_ckpt_format = False
    args.exit_on_missing_checkpoint = False
    args.finetune = False
    args.consumed_train_samples = 0
    args.skipped_train_samples = 0
    args.consumed_valid_samples = 0
    args.no_load_optim = False
    args.no_load_rng = False
    args.dist_ckpt_strictness = 'assume_ok_unexpected'
    args.add_position_embedding = True
    args.vocab_file = False
    args.num_layers = NUM_LAYERS
    args.hidden_size = HIDDEN_SIZE
    args.num_attention_heads = NUM_ATTENTION_HEADS
    args.ckpt_step = None
    args.use_megatron_fsdp = False
    args.dist_ckpt_optim_fully_reshardable = False
    args.distrib_optim_fully_reshardable_mem_efficient = False
    args.phase_transition_iterations = None
    # Clear the metadata cache to avoid contamination between tests

    CachedMetadataFileSystemReader.clear_metadata_cache()


def setup_model_and_optimizer(
    seed,
    tp,
    pp,
    initialize_fn=initialize_gpt_model,
    bf16=True,
    dist_opt=True,
    optimizer='adam',
    use_param_layout=False,
):
    optimizer_type = optimizer
    use_layer_wise = False
    if optimizer_type == 'dist_muon':
        optimizer = 'muon'
        use_layer_wise = True
    if optimizer_type in ('muon', 'dist_muon') and dist_opt:
        use_layer_wise = True

    # When use_layer_wise is True and use_param_layout is False, route DDP
    # construction through the legacy path (no precomputed param layout, no
    # ``use_distributed_optimizer=True`` flip). LayerWiseDistributedOptimizer
    # then syncs via its legacy ``allgather_params()`` codepath rather than
    # ``start_param_sync``.
    ddp_use_dist_opt = dist_opt and not (use_layer_wise and not use_param_layout)
    ddp_use_layer_wise = use_layer_wise and use_param_layout

    mock_args = parse_args(ignore_unknown_args=True)
    with mock.patch('megatron.training.training.get_args', new=lambda: mock_args):
        init_basic_mock_args(mock_args, tp, pp, bf16=bf16)
        mock_args.use_distributed_optimizer = ddp_use_dist_opt
        mock_args.use_layer_wise_distributed_optimizer = ddp_use_layer_wise
        if ddp_use_layer_wise:
            mock_args.optimizer = optimizer
        model = get_model(
            partial(
                initialize_fn,
                seed=seed,
                tensor_model_parallel_size=tp,
                pipeline_model_parallel_size=pp,
                pipeline_dtype=torch.bfloat16,
                bf16=bf16,
            )
        )

    config = OptimizerConfig(
        bf16=bf16,
        params_dtype=torch.bfloat16 if bf16 else torch.float,
        use_distributed_optimizer=ddp_use_dist_opt,
        use_layer_wise_distributed_optimizer=use_layer_wise,
        optimizer=optimizer,
    )

    if optimizer_type in ('muon', 'dist_muon'):
        config.lr = 0.0
    optimizer = get_megatron_optimizer(config, model)

    torch.manual_seed(seed + 1)
    model_parallel_cuda_manual_seed(seed + 1)

    def _init_states(optimizer):
        # In hybrid LayerWise + DistOpt mode the top-level ChainedOptimizer
        # wraps another ChainedOptimizer (LayerWise) alongside DistOpt; recurse
        # so the Muon Float16 sub-optimizers inside LayerWise still get their
        # state seeded. Optimizers without ``init_state_fn`` (DistOpt) seed
        # their state elsewhere and are skipped here.
        if isinstance(optimizer, ChainedOptimizer):
            for child_optimizer in optimizer.chained_optimizers:
                _init_states(child_optimizer)
            return
        if not hasattr(optimizer, 'init_state_fn'):
            return
        if not hasattr(optimizer, 'optimizer'):
            optimizer.init_state_fn(optimizer)
        else:
            optimizer.init_state_fn(optimizer.optimizer)

    if isinstance(optimizer, ChainedOptimizer):
        _init_states(optimizer)
    else:
        for group in optimizer.optimizer.param_groups:
            for p in group['params']:
                if len(optimizer.optimizer.state[p]) == 0:
                    optimizer.optimizer.state[p]['exp_avg'] = torch.rand_like(p.data)
                    optimizer.optimizer.state[p]['exp_avg_sq'] = torch.rand_like(p.data)

    optimizer.reload_model_params()
    CachedMetadataFileSystemReader.clear_metadata_cache()
    return unwrap_model(model), optimizer


def find_matching_values(
    x: Union[dict, list], predicate: Callable[[Any], bool]
) -> Tuple[Union[dict, list], Union[dict, list]]:
    """Return matching values in a single list

    Args:
        x (Union[dict, list]) : state dict to process. Top-level argument must be a dict or list
        predicate (object -> bool): determines matching values
    """

    matching_vals = []
    if hasattr(x, 'values') and callable(getattr(x, 'values')):
        values = x.values()
    elif isinstance(x, list):
        values = x
    else:
        raise ValueError(f'Unexpected top-level object type: {type(x)}')
    for v in values:
        if isinstance(v, (list, dict)):
            matching_vals += find_matching_values(v, predicate)
        elif predicate(v):
            matching_vals.append(v)
    return matching_vals


def setup_moe_model_and_optimizer(
    seed,
    tp,
    pp,
    ep,
    initialize_fn=initialize_moe_model,
    bf16=True,
    dist_opt=True,
    use_te=False,
    use_grouped_mlp=False,
    use_glu=False,
    optimizer='adam',
    use_param_layout=False,
):
    optimizer_type = optimizer
    use_layer_wise = False
    if optimizer_type == 'dist_muon':
        optimizer = 'muon'
        use_layer_wise = True
    if optimizer_type in ('muon', 'dist_muon') and dist_opt:
        use_layer_wise = True

    # See setup_model_and_optimizer for the use_param_layout semantics.
    ddp_use_dist_opt = dist_opt and not (use_layer_wise and not use_param_layout)
    ddp_use_layer_wise = use_layer_wise and use_param_layout

    mock_args = parse_args(ignore_unknown_args=True)
    with mock.patch('megatron.training.training.get_args', new=lambda: mock_args):
        init_basic_mock_args(mock_args, tp, pp, bf16=bf16)
        mock_args.use_distributed_optimizer = ddp_use_dist_opt
        mock_args.use_layer_wise_distributed_optimizer = ddp_use_layer_wise
        if ddp_use_layer_wise:
            mock_args.optimizer = optimizer
        model = get_model(
            partial(
                initialize_fn,
                seed=seed,
                tensor_model_parallel_size=tp,
                pipeline_model_parallel_size=pp,
                pipeline_dtype=torch.bfloat16,
                expert_model_parallel_size=ep,
                use_sp=(tp > 1 and ep > 1),
                use_te=use_te,
                use_grouped_mlp=use_grouped_mlp,
                use_glu=use_glu,
                bf16=bf16,
            )
        )

    config = OptimizerConfig(
        bf16=bf16,
        params_dtype=torch.bfloat16 if bf16 else torch.float,
        use_distributed_optimizer=ddp_use_dist_opt,
        use_layer_wise_distributed_optimizer=use_layer_wise,
        optimizer=optimizer,
    )

    if optimizer_type in ('muon', 'dist_muon'):
        config.lr = 0.0
    optimizer = get_megatron_optimizer(config, model)

    torch.manual_seed(seed + 1)
    model_parallel_cuda_manual_seed(seed + 1)

    if optimizer_type in ('muon', 'dist_muon'):
        for opt in optimizer.chained_optimizers:
            if not hasattr(opt, 'optimizer'):
                opt.init_state_fn(opt)
            else:
                opt.init_state_fn(opt.optimizer)
    else:
        for opt in optimizer.chained_optimizers:
            for group in opt.param_groups:
                for p in group['params']:
                    if len(opt.state[p]) == 0:
                        opt.state[p]['exp_avg'] = torch.rand_like(p.data)
                        opt.state[p]['exp_avg_sq'] = torch.rand_like(p.data)

    optimizer.reload_model_params()
    CachedMetadataFileSystemReader.clear_metadata_cache()
    return unwrap_model(model), optimizer
