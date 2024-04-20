# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import os
import torch
from torch import Tensor
from functools import partial
from typing import Union
from megatron import get_args
from megatron import print_rank_0, print_rank_last
from megatron import get_timers
from megatron import get_tokenizer
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import GPTDataset
import megatron.model
from megatron.core.models.gpt import GPTModel
from megatron.training import pretrain
from megatron.core.transformer.spec_utils import import_module
from megatron.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    average_losses_across_data_parallel_group
)
from megatron.arguments import core_transformer_config_from_args
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.initialize import initialize_megatron
from megatron.training import get_model
from megatron.core.utils import get_model_config
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron import get_num_microbatches
from megatron.checkpointing import _load_base_checkpoint


def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.model.GPTModel]:
    """Builds the model.

    If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.model.GPTModel]: The returned model
    """
    args = get_args()

    print_rank_0('building GPT model ...')
    config = core_transformer_config_from_args(get_args())

    if args.use_mcore_models:
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm)

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
            rotary_percent=args.rotary_percent
        )
    else:
        assert(args.context_parallel_size == 1), "Context parallelism is only supported with Megatron Core!"

        model = megatron.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process
        )

    return model


def loss_func(loss_mask: Tensor, output_tensor: Tensor):
    """Loss function.

    Args:
        loss_mask (Tensor): Used to mask out some portions of the loss
        output_tensor (Tensor): The tensor with the losses
    """    
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    if args.context_parallel_size > 1:
        loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), loss_mask.sum().view(1)])
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())
        loss = loss[0] / loss[1]
    else:
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss.isnan(), (
            f'Rank {global_rank}: found NaN in local forward loss calculation. '
            f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
        )

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss * args.context_parallel_size, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()

    # BATCH SIZE PER SINGLE GPU MUST BE ONE!!!!
    tokens = torch.tensor([[    1, 20811,   349,   396, 13126,   369, 13966,   264]], device=torch.cuda.current_device())
    position_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], device=torch.cuda.current_device())
    loss_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]], device=torch.cuda.current_device())
    attention_mask = torch.tensor([[[[False,  True,  True,  True,  True,  True,  True,  True],
              [False, False,  True,  True,  True,  True,  True,  True],
              [False, False, False,  True,  True,  True,  True,  True],
              [False, False, False, False,  True,  True,  True,  True],
              [False, False, False, False, False,  True,  True,  True],
              [False, False, False, False, False, False,  True,  True],
              [False, False, False, False, False, False, False,  True],
              [False, False, False, False, False, False, False, False]]]], device=torch.cuda.current_device())
    #labels = torch.tensor([[32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000]], device=torch.cuda.current_device())
    labels = torch.tensor([[20896, 26570, 20896, 21876, 25931, 25931, 20896, 20896]], device=torch.cuda.current_device())

    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)

    if isinstance(output_tensor, tuple):
        return list(output_tensor), partial(loss_func, loss_mask)
    else:
        return output_tensor, partial(loss_func, loss_mask)


if __name__ == "__main__":
    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(extra_args_provider=None, args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})

    args = get_args()
    args.model_type = ModelType.encoder_or_decoder
    model = get_model(model_provider, ModelType.encoder_or_decoder)
    print_rank_0(model)

    config = get_model_config(model[0])

    if args.load is not None:
        load_dir = getattr(args, 'load')
        state_dict, _, _ = _load_base_checkpoint(load_dir, rank0=False)
        model[0].module.load_state_dict(state_dict['model'], strict=True)

    for model_module in model:
        model_module.eval()
        #model_module.train()

    total_loss_dict = {}

    for model_chunk in model:
        model_chunk.zero_grad_buffer(zero_buffer=(not args.use_distributed_optimizer))

    forward_backward_func = get_forward_backward_func()
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step,
        data_iterator=None,
        model=model,
        num_microbatches=get_num_microbatches(),
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        decoder_seq_length=args.decoder_seq_length,
        forward_only=False)

    #print(model[0].module.module.language_model.encoder.layers[-1].mlp.experts[0].w1.weight)
    #print(model[0].module.module.language_model.encoder.layers[-1].mlp.gate.weight.main_grad)

    #print_rank_last(losses_reduced)
