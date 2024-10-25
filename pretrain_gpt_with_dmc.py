# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""Pretrain GPT with Dynamic Memory Compression (DMC)."""

from megatron.contrib.dmc.arguments import add_dmc_args
add_dmc_args()

import os
import torch
from contextlib import nullcontext
from functools import partial
import inspect
from typing import Union

import megatron.legacy.model
from megatron.contrib.dmc import add_dmc_layer, get_dmc_loss
from megatron.core import mpu
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec, get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.spec_utils import import_module
from megatron.core.utils import StragglerDetector
from megatron.training import get_args, get_model, get_timers, get_tokenizer, pretrain
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from megatron.training.global_vars import set_args
from megatron.training.initialize import initialize_megatron
from megatron.training.utils import (
    get_batch_on_this_cp_rank, get_batch_on_this_tp_rank, print_rank_0, unwrap_model
)
from megatron.training.yaml_arguments import core_transformer_config_from_yaml


stimer = StragglerDetector()


def model_provider(
    pre_process=True, post_process=True
) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not
    the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings.
            Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output
            logits/loss. Defaults to True.

    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.use_legacy_models:
        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )
    else:  # Using core models
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if use_te:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                    args.num_experts,
                    args.moe_grouped_gemm,
                    args.qk_layernorm
                )
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(
                    args.num_experts,
                    args.moe_grouped_gemm,
                    args.qk_layernorm
                )

        add_dmc_layer(transformer_layer_spec)

        build_model_context = nullcontext
        build_model_context_args = {}
        if args.fp8_param_gather:
            try:
                from transformer_engine.pytorch import fp8_model_init

                build_model_context = fp8_model_init
                build_model_context_args["enabled"] = True

                # Check if fp8_model_init supports preserve_high_precision_init_val
                if "preserve_high_precision_init_val" in inspect.signature(fp8_model_init).parameters:
                    build_model_context_args["preserve_high_precision_init_val"] = True
            except:
                raise RuntimeError("--fp8-param-gather requires `fp8_model_init` from TransformerEngine, but not found.")

        with build_model_context(**build_model_context_args):
            model = GPTModel(
                config=config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=args.padded_vocab_size,
                max_sequence_length=args.max_position_embeddings,
                pre_process=pre_process,
                post_process=post_process,
                fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
                parallel_output=True,
                share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
                position_embedding_type=args.position_embedding_type,
                rotary_percent=args.rotary_percent,
                rotary_base=args.rotary_base,
                rope_scaling=args.use_rope_scaling
            )

    return model


def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None

    # Get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)

    # Slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    return batch.values()


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
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

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    total_tokens = loss_mask.sum()
    loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])

    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss[0].isnan(), (
            f'Rank {global_rank}: found NaN in local forward loss calculation. '
            f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
        )

    reporting_loss = loss.clone().detach()
    torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())
    local_num_tokens = loss[1].clone().detach().to(torch.int)

    if args.dmc_is_stage_one:
        return (
            loss[0] * args.context_parallel_size,
            local_num_tokens,
            {'lm loss': (reporting_loss[0], reporting_loss[1])},
        )

    dmc_loss, dmc_curr_cr, dmc_tgt_cr = get_dmc_loss()

    # Reduce loss for logging.

    reporting_dmc_loss = dmc_loss.clone().detach()
    torch.distributed.all_reduce(reporting_dmc_loss, group=mpu.get_data_parallel_group())
    reporting_dmc_loss /= torch.distributed.get_world_size(group=mpu.get_data_parallel_group())

    torch.distributed.all_reduce(dmc_curr_cr, group=mpu.get_data_parallel_group())
    dmc_curr_cr /= torch.distributed.get_world_size(group=mpu.get_data_parallel_group())
    return (
        (loss[0] + dmc_loss * local_num_tokens) * args.context_parallel_size,
        local_num_tokens,
        {
            'lm loss': (reporting_loss[0], reporting_loss[1]),
            'dmc loss': reporting_dmc_loss,
            'dmc curr cr': dmc_curr_cr,
            'dmc tgt cr': dmc_tgt_cr,
        },
    )


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    timers('batch-generator').stop()

    with stimer:
        output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def is_dataset_built_on_rank():
    return (
        mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
    ) and mpu.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=get_blend_from_list(args.data_path),
        blend_per_split=[
            get_blend_from_list(args.train_data_path),
            get_blend_from_list(args.valid_data_path),
            get_blend_from_list(args.test_data_path)
        ],
        renormalize_blend_weights=args.renormalize_blend_weights,
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        s3_cache_path=args.s3_cache_path
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test
            and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    if args.mock_data:
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        is_dataset_built_on_rank,
        config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


def save_stage_one_final_model(execute_independently=False):
    if not get_args().dmc_is_stage_one:
        print_rank_0("Not modifying checkpoint because DMC training stage is 2 or higher")
        return
    print_rank_0(
        "Neuron ablation steps have finished. Modifying checkpoint and saving "
        "once again such that KV channels corresponding to ablated neuron are set to 0..."
    )

    if execute_independently:
        # If no prior pretrain is run in the same execution, we first have to initialize megatron.
        # This assumes that args.load and args.save are already correctly specified
        initialize_megatron(extra_args_provider=None)
        args = get_args()

    else:
        # Overwrite args from previous megatron pretraining to change save path
        args = get_args()
        tmp = args.save
        args.load = None
        args.pretrained_checkpoint = args.save
        args.save = os.path.join(tmp, 'final_output')
        args.consumed_train_samples = 0
        args.consumed_valid_samples = 0
        print_rank_0(f"modfied save path for modified checkpoint from {tmp} to {args.save}")
        set_args(args)

    model = get_model(model_provider, ModelType.encoder_or_decoder)
    unwrapped_model = unwrap_model(model)

    _ = load_checkpoint(model, None, None)

    # Set kv channels corresponding to ablated neuron to zero
    print_rank_0("setting query channels corresponding to ablated neuron to zero")
    kv = args.kv_channels
    for i in range(len(unwrapped_model[0].decoder.layers)):
        layer = unwrapped_model[0].decoder.layers[i].self_attention
        XX = layer.linear_qkv.weight.reshape(
            layer.num_query_groups_per_partition,
            (layer.num_attention_heads_per_partition // layer.num_query_groups_per_partition + 2)
            * layer.hidden_size_per_attention_head,
            -1,
        )
        torch.nn.init.zeros_(XX[:, kv - 1:-kv:kv, :])

    print_rank_0(f"saving model to {args.save}")
    save_checkpoint(
        1,
        model,
        optimizer=None,
        opt_param_scheduler=None,
        num_floating_point_operations_so_far=0,
        checkpointing_context=None,
    )


if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
    )

    if get_args().dmc_is_stage_one:
        save_stage_one_final_model()
