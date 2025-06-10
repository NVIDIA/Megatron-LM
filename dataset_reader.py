# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
"""Data reader."""

import json
import numpy as np
import os
import time
import torch
from typing import List, Optional, Tuple, Union

from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_tokenizer
from megatron.training import initialize_megatron
from megatron.core import mpu
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.training.training import build_train_valid_test_data_iterators
from megatron.training.training import update_train_iters
from megatron.core.transformer.spec_utils import import_module
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
)


def is_dataset_built_on_rank():
    return (
        mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
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


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
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


if __name__ == "__main__":

    # Temporary for transition to core datasets.
    train_valid_test_datasets_provider.is_distributed = True

    # Initialize Megatron.
    initialize_megatron()

    # Make sure only data parallelism is used, and patch args before building data iterators.
    args = get_args()
    update_train_iters(args)
    assert mpu.get_data_parallel_world_size() == args.world_size
    args.iteration = 800000  # TODO: Make this configurable from command line?
    batch_size = mpu.get_data_parallel_world_size() * \
                 args.micro_batch_size * \
                 get_num_microbatches()
    args.consumed_train_samples = args.iteration * batch_size  # Assume no batch size rampup.
    args.consumed_valid_samples = 0

    # Build iterators.
    train_data_iterator, _, _ = build_train_valid_test_data_iterators(
        train_valid_test_datasets_provider)

    # Iterate and find problematic tokens.
    problematic_tokens = torch.tensor(
       [  1192,   1193,   1245,   1246,   1247,   1248,   1249,   1250,   1251,   1252,
          1253,   1254,   1255,  25304,  42819,  51067,  67044,  77579,  81422,  81695,
         81772,  81815,  81819,  82267,  82312,  82690,  82725,  82737,  82977,  83855,
         84006,  84105,  84121,  84794,  85015,  86060,  86366,  86409,  87020,  87411,
         89412,  90101,  90294,  90320,  91368,  91515,  91608,  92529,  94485,  98573,
        104097, 106921, 108262, 112277, 112327, 112497, 114442, 114755, 116654, 117744],
       dtype=torch.int32
    ).cuda()
    torch.distributed.barrier()
    start_time = time.time()
    for iteration in range(args.iteration, args.train_iters):
        for _ in range(get_num_microbatches()):
            data = next(train_data_iterator)
            tokens = data["tokens"].cuda()
            mask = torch.isin(problematic_tokens, tokens)
            if len(problematic_tokens[mask]) > 0:
                print(f"Found tokens {problematic_tokens[mask]} in iteration {iteration + 1}")

        if (iteration + 1) % 1000 == 0:
            torch.distributed.barrier()
            elapsed_time = time.time() - start_time
            print_rank_0(f"Iteration: {iteration + 1}/{args.train_iters}, Elapsed time: {elapsed_time:.2f} seconds")

    # Destroy mpu.
    mpu.destroy_model_parallel()
