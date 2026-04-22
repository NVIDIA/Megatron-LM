# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain and SFT Mamba."""

import os
from functools import partial
from typing import List, Optional, Tuple, Union

import torch

from megatron.core import mpu, parallel_state
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.enums import ModelType
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.num_microbatches_calculator import (
    get_current_global_batch_size,
    get_micro_batch_size,
)
from megatron.core.parallel_state import (
    get_context_parallel_rank,
    get_context_parallel_world_size,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
)
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.spec_utils import import_module
from megatron.core.utils import StragglerDetector
from megatron.elastification.arguments import add_flextron_args
from megatron.training import (
    get_args,
    get_timers,
    get_tokenizer,
    inprocess_restart,
    pretrain,
    print_rank_0,
)
from megatron.training.arguments import core_transformer_config_from_args, parse_and_validate_args
from megatron.training.datasets.sft_dataset import SFTDataset
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
)

# modelopt distillation
try:
    from megatron.elastification.loss_func import loss_func as loss_func_modelopt
    from megatron.post_training.arguments import add_modelopt_args
    from megatron.post_training.model_builder import (
        modelopt_gpt_mamba_builder as model_provider_modelopt,
    )
    has_nvidia_modelopt = True
except ImportError:
    raise ImportError("ModelOpt is not installed. Please install it using `pip install nvidia-modelopt`")
    has_nvidia_modelopt = False
print("has_nvidia_modelopt is {}".format(has_nvidia_modelopt))
import numpy as np

try:
    # Register the TE CUDA kernels
    import transformer_engine  # pylint: disable=unused-import

    # Alias the PyTorch wrapper so we can call tex.* APIs
    import transformer_engine_torch as tex
except ImportError:
    # TE isn’t installed or the torch wrapper is missing
    tex = None

from megatron.core.utils import StragglerDetector, is_te_min_version

_global_choice_counter = 0
_logged_params_norm = False

stimer = StragglerDetector()

def count_parameters_in_layer(model, layer_name):
    num_params = 0
    for name, param in model.named_parameters():
        if layer_name in name:
            num_params += param.numel()
            print_rank_0(f" - {name}: {param.numel()}")
    return num_params


def model_provider(pre_process=True, post_process=True, vp_stage: Optional[int] = None, config = None, pg_collection = None) -> HybridModel:
    """Builds the model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        HybridModel: The returned model
    """
    args = get_args()
    if has_nvidia_modelopt:

        model = model_provider_modelopt(args, pre_process, post_process, vp_stage=vp_stage, config=config, pg_collection=pg_collection)
        from megatron.elastification.flextron_utils import (
            inject_flextron_forward_logic,
            setup_flextron_model,
        )
        setup_flextron_model(model)
        inject_flextron_forward_logic(model)

        if args.freeze_model:
            for name, param in model.named_parameters(): 
                if 'gate' not in name:
                    param.requires_grad = False
                    
        if args.freeze_router:
            for name, param in model.named_parameters(): 
                if 'gate' in name:
                    param.requires_grad = False

        return model

    print_rank_0('building Mamba model ...')
    config = core_transformer_config_from_args(args, TransformerConfig)

    assert args.use_legacy_models == False, "Mamba only supported in Mcore!"

    if args.spec is not None:
        hybrid_stack_spec = import_module(args.spec)
    else:
        raise("You must provide a valid Mamba layer spec!")

    model = HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        hybrid_layer_pattern=args.hybrid_layer_pattern,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        vp_stage=vp_stage
    )
    from megatron.elastification.flextron_utils import (
        inject_flextron_forward_logic,
        setup_flextron_model,
    )
    setup_flextron_model(model)
    inject_flextron_forward_logic(model)

    for l in range(model.decoder.num_layers_per_pipeline_rank):
        layer_params = count_parameters_in_layer(model, f'decoder.layers.{l}.')
        print_rank_0(f" == params layer {l}: {layer_params}")

    return model


def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)

    cu_seqlens = batch['cu_seqlens']
    if cu_seqlens is None:
        # slice batch along sequence dimension for context parallelism
        batch = get_batch_on_this_cp_rank(batch)  # The implementation of this function is in MCore
    else:  # Packed THD format
        assert (
            cu_seqlens.dim() == 2 and cu_seqlens.shape[0] == 1
        ), "micro-batch-size must be 1 for packing"
        cu_seqlens = cu_seqlens[0]
        batch['cu_seqlens'] = cu_seqlens

        max_seqlen = batch['max_seqlen']
        assert max_seqlen.dim() == 1
        # TODO(duncan): can this be kept as a 0-D tensor?
        batch['max_seqlen'] = int(max_seqlen[0].item())

        cp_size = get_context_parallel_world_size()
        if cp_size > 1:  # slice batch along sequence dimension for context parallelism
            assert tex is not None and is_te_min_version("1.10.0"), (
                "Please update Transformer Engine to >= 1.10 to use "
                "Context Parallel with THD format data"
            )
            cp_rank = get_context_parallel_rank()
            index = tex.thd_get_partitioned_indices(
                cu_seqlens,
                batch['tokens'].size(1),
                cp_size,
                cp_rank,
            )
            for key, data in batch.items():
                if key in {'attention_mask', 'cu_seqlens', 'max_seqlen'}:
                    continue
                batch[key] = data.index_select(1, index)

    return (
        batch.get('tokens'),
        batch.get('labels'),
        batch.get('loss_mask'),
        batch.get('attention_mask'),
        batch.get('position_ids'),
        batch.get('cu_seqlens'),
        batch.get('max_seqlen'),
    )


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: Optional[HybridModel] = None, selected_budget=None):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()
    if has_nvidia_modelopt:
        return loss_func_modelopt(loss_mask, output_tensor, model=model, selected_budget=selected_budget)

    alpha = args.loss_alpha 

    (output_tensor, (param_loss, extra_reporting_dict)) = output_tensor
    
    if param_loss is not None:
        if param_loss > 0:
            param_loss_report = param_loss.detach().clone()
        else:
            param_loss_report = param_loss.detach().clone()
            param_loss = -args.router_beta * param_loss

    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * loss_mask)

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=False,
        )


    num_tokens = loss_mask.sum().clone().detach().to(torch.int)

    if param_loss is not None:
        param_loss *= num_tokens * alpha
        if param_loss < 0:
            param_loss = -args.router_beta * param_loss 

        param_loss_report = torch.cat([param_loss.clone().detach().view(1), num_tokens.view(1)])
        lm_loss_report = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])
        loss += param_loss[0]
            
    # Protect against division by zero when all tokens are masked.
    num_tokens = torch.clamp(num_tokens, min=1)
    reporting_loss = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])

    if param_loss is not None:
        return (loss, num_tokens, {'lm loss': lm_loss_report, 
                                   'param loss': param_loss_report,
                                   'total loss': reporting_loss})
    else:
        return (loss, num_tokens, {'lm loss': reporting_loss})

def get_grad_acc_based_random_choice(args, choices=None, prob=None, base_seed=42):

    dp_size = get_data_parallel_world_size()
    grad_accumulation_steps = get_current_global_batch_size() // (get_micro_batch_size() * dp_size)
    global _global_choice_counter

    # DP-specific seeding
    rng = np.random.RandomState(base_seed + _global_choice_counter + grad_accumulation_steps*args.curr_iteration*10)
    if choices is None:
        choice = rng.uniform(0, 1)
    else:
        if prob is None:
            choice = rng.choice(choices)
        else:
            choice = rng.choice(choices, p=prob)
    _global_choice_counter += 1
    _global_choice_counter %= grad_accumulation_steps
    return choice

def forward_step(data_iterator, model: HybridModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (HybridModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()

    # One-time per-component params-norm breakdown (mirrors calc_params_l2_norm).
    global _logged_params_norm
    if not _logged_params_norm:
        _logged_params_norm = True
        from collections import defaultdict
        groups = defaultdict(float)
        trainable_sq = frozen_sq = total_sq = 0.0
        for name, param in model.named_parameters():
            # Use fp32 main_param when distributed optimizer is active (matches WandB metric).
            p = getattr(param, 'main_param', param.detach()).float()
            norm_sq = p.norm(2).item() ** 2
            # Strip DDP 'module.' wrappers to get the logical top-level name.
            clean = name
            while clean.startswith('module.'):
                clean = clean[len('module.'):]
            top = clean.split('.')[0]
            groups[top] += norm_sq
            total_sq += norm_sq
            if param.requires_grad:
                trainable_sq += norm_sq
            else:
                frozen_sq += norm_sq
        print_rank_0(
            f"[PARAMS_NORM] total={total_sq**0.5:.2f}  "
            f"trainable={trainable_sq**0.5:.2f}  frozen={frozen_sq**0.5:.2f}"
        )
        for grp, sq in sorted(groups.items()):
            print_rank_0(f"[PARAMS_NORM]   {grp}: {sq**0.5:.2f}")

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        (
            tokens,
            labels,
            loss_mask,
            attention_mask,
            position_ids,
            cu_seqlens,
            max_seqlen,
        ) = get_batch(data_iterator)
    timers('batch-generator').stop()

    if get_grad_acc_based_random_choice(args=args) < args.original_model_sample_prob:
        flextron_kwargs = {}
        selected_budget = 1.0  # Full model
    else:
        if args.budget_probs is None:
            budget_probs = [1.0 for _ in args.budget_list]
        else:
            budget_probs = args.budget_probs
            
        assert len(args.budget_list) == len(budget_probs), "budget_list and budget_probs must have the same length"
        budget_probs = [float(p) for p in budget_probs]
        budget_probs = [p / sum(budget_probs) for p in budget_probs]
        selected_budget = get_grad_acc_based_random_choice(args=args, choices=args.budget_list, prob=budget_probs)
        flextron_kwargs = {'budget': selected_budget}

    with stimer:
        output_tensor = model(tokens, position_ids, attention_mask,
                              labels=labels, **flextron_kwargs)

    # [ModelOpt]: model is needed to access ModelOpt distillation losses
    return output_tensor, partial(loss_func, loss_mask, model=model, selected_budget=selected_budget)



def is_dataset_built_on_rank(vp_stage=None):
    ignore_virtual = True
    if vp_stage is not None:
        ignore_virtual = False
    return (
        mpu.is_pipeline_first_stage(ignore_virtual=ignore_virtual, vp_stage=vp_stage)
        or mpu.is_pipeline_last_stage(ignore_virtual=ignore_virtual, vp_stage=vp_stage)
    ) and mpu.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    # Sometimes --data-path is too long, instead we parse it from a file.
    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        object_storage_cache_path=args.object_storage_cache_path,
        mid_level_dataset_surplus=args.mid_level_dataset_surplus,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    if args.sft:
        dataset_type = SFTDataset
    else:
        if args.mock_data:
            dataset_type = MockGPTDataset
        else:
            dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        partial(is_dataset_built_on_rank, vp_stage=vp_stage),
        config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


def mamba_flex_extra_args_provider(parser):
    """Add Flextron CLI if not already registered by ``add_megatron_arguments``, then ModelOpt."""
    if not any(getattr(action, "dest", None) == "flextron" for action in parser._actions):
        parser = add_flextron_args(parser)
    if has_nvidia_modelopt:
        parser = add_modelopt_args(parser)
    return parser


if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    # Optionally enable inprocess restart on pretrain
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    # Restore router LR multiplier (Bug 4 fix): monkey-patch get_megatron_optimizer_config
    # to inject a per-parameter LR override for router params via config_overrides.
    # Main branch removed the scale_lr_cond parameter from pretrain(); this achieves the same.
    import megatron.training.training as _mtt
    from megatron.core.optimizer.optimizer_config import ParamKey, ParamWithNamePredicate
    from megatron.core.optimizer_param_scheduler import ParamGroupOverride

    _orig_get_opt_cfg = _mtt.get_megatron_optimizer_config

    def _patched_get_opt_cfg(args):
        config, config_overrides = _orig_get_opt_cfg(args)
        lr_mult = getattr(args, 'lr_mult_router', 1.0)
        if lr_mult != 1.0:
            router_key = ParamKey(
                with_name_predicate=ParamWithNamePredicate(
                    name="router_pp",
                    fn=lambda p, name: 'router_pp' in name,
                )
            )
            router_override = ParamGroupOverride(
                max_lr=args.lr * lr_mult,
                min_lr=args.min_lr * lr_mult,
            )
            config_overrides = {**(config_overrides or {}), router_key: router_override}
        return config, config_overrides

    _mtt.get_megatron_optimizer_config = _patched_get_opt_cfg

    # `pretrain()` no longer accepts extra_args_provider / args_defaults; parse
    # args up-front instead (see pretrain_mamba.py for the same pattern).
    args = parse_and_validate_args(
        extra_args_provider=mamba_flex_extra_args_provider,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )

    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             store=store,
             )
