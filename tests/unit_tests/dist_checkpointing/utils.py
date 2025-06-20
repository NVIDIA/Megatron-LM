from functools import partial
from typing import Any, Callable, Tuple, Union
from unittest import mock

import torch

from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
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
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)

    default_config_kwargs = dict(
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_ATTENTION_HEADS,
        use_cpu_initialization=True,
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

    model.bfloat16()
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
    args.log_progress = False
    args.auto_detect_ckpt_format = False
    args.exit_on_missing_checkpoint = False
    args.finetune = False
    args.consumed_train_samples = 0
    args.skipped_train_samples = 0
    args.consumed_valid_samples = 0
    args.retro_add_retriever = False
    args.no_load_optim = False
    args.no_load_rng = False
    args.dist_ckpt_strictness = 'assume_ok_unexpected'
    args.add_position_embedding = True
    args.vocab_file = False
    args.num_layers = NUM_LAYERS
    args.hidden_size = HIDDEN_SIZE
    args.num_attention_heads = NUM_ATTENTION_HEADS
    args.ckpt_step = None


def setup_model_and_optimizer(
    seed,
    tp,
    pp,
    initialize_fn=initialize_gpt_model,
    bf16=True,
    dist_opt=True,
    use_custom_fsdp=False,
    data_parallel_sharding_strategy="optim_grads_params",
):
    mock_args = parse_args(ignore_unknown_args=True)
    with mock.patch('megatron.training.training.get_args', new=lambda: mock_args):
        init_basic_mock_args(mock_args, tp, pp, bf16=bf16)
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
        use_distributed_optimizer=dist_opt,
    )
    optimizer = get_megatron_optimizer(config, model)

    torch.manual_seed(seed + 1)
    model_parallel_cuda_manual_seed(seed + 1)

    for group in optimizer.optimizer.param_groups:
        for p in group['params']:
            if len(optimizer.optimizer.state[p]) == 0:
                optimizer.optimizer.state[p]['exp_avg'] = torch.rand_like(p.data)
                optimizer.optimizer.state[p]['exp_avg_sq'] = torch.rand_like(p.data)

    optimizer.reload_model_params()

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
):
    mock_args = parse_args(ignore_unknown_args=True)
    with mock.patch('megatron.training.training.get_args', new=lambda: mock_args):
        init_basic_mock_args(mock_args, tp, pp, bf16=bf16)
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
        use_distributed_optimizer=dist_opt,
    )
    optimizer = get_megatron_optimizer(config, model)

    torch.manual_seed(seed + 1)
    model_parallel_cuda_manual_seed(seed + 1)

    for opt in optimizer.chained_optimizers:
        for group in opt.param_groups:
            for p in group['params']:
                if len(opt.state[p]) == 0:
                    opt.state[p]['exp_avg'] = torch.rand_like(p.data)
                    opt.state[p]['exp_avg_sq'] = torch.rand_like(p.data)

    optimizer.reload_model_params()

    return unwrap_model(model), optimizer
