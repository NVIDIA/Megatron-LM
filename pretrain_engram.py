# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain Engram-augmented GPT model."""

import time
_PROGRAM_START_TIME = time.time()

import json
import os
import warnings

rank = int(os.environ.get('RANK', 0))
if rank != 0:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

from functools import partial
from typing import List, Optional, Tuple

import torch

from engram_builders import engram_builder
from megatron.core import mpu
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.enums import ModelType
from megatron.core.models.engram import EngramGPTModel
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.core.utils import get_attr_wrapped_model, StragglerDetector
from megatron.training import (
    get_args,
    get_timers,
    inprocess_restart,
    pretrain,
    print_rank_0,
    set_startup_timestamps,
)
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
    is_first_or_last_pipeline_stage,
)
from model_provider import model_provider


stimer = StragglerDetector()


def get_batch(data_iterator, vp_stage=None):
    """Generate a batch."""
    empty_batch = {
        'tokens': None,
        'labels': None,
        'loss_mask': None,
        'attention_mask': None,
        'position_ids': None,
    }

    if not is_first_or_last_pipeline_stage(vp_stage):
        return empty_batch.values()

    batch = get_batch_on_this_tp_rank(data_iterator)

    if mpu.is_pipeline_first_stage(ignore_virtual=(vp_stage is None), vp_stage=vp_stage):
        total_tokens = batch['tokens'].size(1)
    elif mpu.is_pipeline_last_stage(ignore_virtual=(vp_stage is None), vp_stage=vp_stage):
        total_tokens = batch['labels'].size(1)
    else:
        return empty_batch.values()

    batch = get_batch_on_this_cp_rank(batch)

    return batch.values()


SPIKY_LOSS_FACTOR = 10


def loss_func(
    loss_mask: torch.Tensor,
    output_tensor: torch.Tensor,
    model: Optional[EngramGPTModel] = None,
):
    """Loss function for Engram GPT training."""
    args = get_args()

    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * loss_mask)

    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    report = {'lm loss': torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])}

    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,
            fatal=True,
        )
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,
            fatal=False,
        )

    return loss, num_tokens, report


def forward_step(data_iterator, model: EngramGPTModel):
    """Forward training step."""
    timers = get_timers()

    timers('batch-generator', log_level=2).start()

    global stimer

    with stimer(bdata=True):
        vp_stage = get_attr_wrapped_model(model, "vp_stage")
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
            data_iterator, vp_stage
        )

    timers('batch-generator').stop()

    with stimer:
        output_tensor = model(
            tokens,
            position_ids,
            attention_mask,
            labels=labels,
            loss_mask=loss_mask,
        )

    return output_tensor, partial(loss_func, loss_mask, model=model)


def is_dataset_built_on_rank(vp_stage=None):
    if mpu.get_tensor_model_parallel_rank() != 0:
        return False
    return is_first_or_last_pipeline_stage(vp_stage)


def core_gpt_dataset_config_from_args(args):
    tokenizer = build_tokenizer(args)

    blend, blend_per_split = get_blend_and_blend_per_split(args)

    sequences_per_dataset = None
    if args.per_dataset_sequences_path is not None:
        with open(args.per_dataset_sequences_path, "r") as f:
            sequences_per_dataset = json.load(f)

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
        allow_ambiguous_pad_tokens=args.allow_ambiguous_pad_tokens,
        fast_cache_load=args.dataloader_fast_cache_load,
        sequences_per_dataset=sequences_per_dataset,
        defer_npy_index_mmap=args.dataloader_defer_npy_index_mmap,
        context_parallel_size=args.context_parallel_size,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    """Build the train, validation, and test datasets."""
    args = get_args()
    config = core_gpt_dataset_config_from_args(args)

    if args.mock_data:
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for Engram GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        partial(is_dataset_built_on_rank, vp_stage=vp_stage),
        config,
    ).build()

    print_rank_0("> finished creating Engram GPT datasets ...")

    return train_ds, valid_ds, test_ds


def add_engram_args(parser):
    """Add Engram-specific command line arguments."""
    group = parser.add_argument_group(title='Engram')
    group.add_argument(
        '--engram-layer-ids',
        type=str,
        default='1,15',
        help='Comma-separated list of 1-based layer IDs that get Engram modules.',
    )
    group.add_argument(
        '--engram-max-ngram-size',
        type=int,
        default=3,
        help='Maximum n-gram size for Engram hashing.',
    )
    group.add_argument(
        '--engram-n-embed-per-ngram',
        type=int,
        default=512,
        help='Embedding dimension per n-gram level in Engram.',
    )
    group.add_argument(
        '--engram-n-head-per-ngram',
        type=int,
        default=8,
        help='Number of hash heads per n-gram level.',
    )
    group.add_argument(
        '--engram-kernel-size',
        type=int,
        default=4,
        help='Kernel size for Engram short convolution.',
    )
    group.add_argument(
        '--engram-hc-mult',
        type=int,
        default=4,
        help='Hyper-connection multiplier for Engram gating.',
    )
    group.add_argument(
        '--engram-pad-id',
        type=int,
        default=2,
        help='Pad token ID for Engram hash computation.',
    )
    group.add_argument(
        '--engram-seed',
        type=int,
        default=0,
        help='Random seed for Engram hash multiplier generation.',
    )
    group.add_argument(
        '--engram-tokenizer',
        type=str,
        default='deepseek-ai/DeepSeek-V3',
        help='Tokenizer name/path for Engram compressed tokenizer.',
    )
    return parser


if __name__ == "__main__":
    _MAIN_ENTRY_TIME = time.time()
    set_startup_timestamps(program_start=_PROGRAM_START_TIME, main_entry=_MAIN_ENTRY_TIME)

    train_valid_test_datasets_provider.is_distributed = True

    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    pretrain(
        train_valid_test_datasets_provider,
        partial(model_provider, engram_builder),
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        store=store,
        extra_args_provider=add_engram_args,
    )
