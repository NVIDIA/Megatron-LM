# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""BAGEL ``PackedDataset`` provider aligned with the native training recipe."""

import copy
import os
import random
from functools import partial

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from examples.mimo_bagel.configs.reference import get_reference_data_seed
from megatron.training import get_args
from megatron.training.utils import print_rank_0


def _build_data_config_kwargs(args, vae_image_downsample):
    """Build the exact ``DataConfig`` values used by native BAGEL training."""

    return {
        'text_cond_dropout_prob': getattr(args, 'text_cond_dropout_prob', 0.1),
        'vit_cond_dropout_prob': getattr(args, 'vit_cond_dropout_prob', 0.3),
        'vae_cond_dropout_prob': getattr(args, 'vae_cond_dropout_prob', 0.3),
        'vae_image_downsample': vae_image_downsample,
        'max_latent_size': getattr(args, 'max_latent_size', 64),
        'vit_patch_size': getattr(args, 'vit_patch_size', 14),
        'max_num_patch_per_side': getattr(args, 'max_num_patch_per_side', 70),
    }


def _build_packed_dataset_kwargs(
    args, *, data_config, tokenizer, special_tokens, data_parallel_rank, data_parallel_world_size
):
    """Build packer kwargs without conflating its yield and padded lengths."""

    packing_buffer_size = getattr(args, 'packing_buffer_size', None)
    if packing_buffer_size is None:
        packing_buffer_size = 50

    return {
        'data_config': data_config,
        'tokenizer': tokenizer,
        'special_tokens': special_tokens,
        'local_rank': data_parallel_rank,
        'world_size': data_parallel_world_size,
        'num_workers': getattr(args, 'num_workers', 1),
        'expected_num_tokens': getattr(args, 'expected_num_tokens', 32768),
        'max_num_tokens_per_sample': getattr(args, 'max_num_tokens_per_sample', 16384),
        'max_num_tokens': getattr(args, 'max_num_tokens', 36864),
        'prefer_buffer_before': getattr(args, 'prefer_buffer_before', 16384),
        'max_buffer_size': packing_buffer_size,
        'interpolate_pos': getattr(args, 'interpolate_pos', False),
        'use_flex': getattr(args, 'use_flex_attention', False),
    }


def _seed_bagel_worker(worker_id: int, *, rank_seed: int) -> None:
    """Isolate data RNG from model construction and prefetch timing."""

    worker_seed = rank_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**32))
    torch.manual_seed(worker_seed)


def _build_data_loader(dataset, args, rank_seed):
    from bagel.data.dataset_base import collate_wrapper

    num_workers = getattr(args, 'num_workers', 1)
    loader_kwargs = {
        'dataset': dataset,
        'batch_size': 1,
        'collate_fn': collate_wrapper(),
        'num_workers': num_workers,
        'pin_memory': True,
        'drop_last': True,
    }
    if num_workers > 0:
        generator = torch.Generator()
        generator.manual_seed(rank_seed)
        loader_kwargs.update(
            prefetch_factor=getattr(args, 'prefetch_factor', 2),
            generator=generator,
            worker_init_fn=partial(_seed_bagel_worker, rank_seed=rank_seed),
        )
    else:
        # The aligned recipe uses one worker. Keep the data-only fallback
        # deterministic too, while making its global-RNG behavior explicit.
        _seed_bagel_worker(0, rank_seed=rank_seed)
    return DataLoader(**loader_kwargs)


def bagel_dataloader_provider(train_val_test_num_samples):
    """Build BAGEL train/validation/test iterators."""

    args = get_args()

    from bagel.data.data_utils import add_special_tokens
    from bagel.data.dataset_base import DataConfig, PackedDataset
    from bagel.modeling.qwen2 import Qwen2Tokenizer

    tokenizer = Qwen2Tokenizer.from_pretrained(args.tokenizer_model)
    tokenizer, special_token_ids, num_new_tokens = add_special_tokens(tokenizer)

    if args.rank == 0:
        print(f"Added {num_new_tokens} special tokens to tokenizer")
        print(f"Special token IDs: {special_token_ids}")

    dataset_config_file = os.environ.get("DATA_CONFIG_FILE")
    if not dataset_config_file:
        raise ValueError("DATA_CONFIG_FILE must point to the BAGEL dataset YAML")
    with open(dataset_config_file, "r", encoding="utf-8") as stream:
        dataset_meta = yaml.safe_load(stream)

    vae_image_downsample = getattr(args, 'vae_image_downsample', 16)
    print_rank_0("dataset_meta", dataset_meta)

    # At TP/PP/CP=1 these are exactly native BAGEL's global rank/world size.
    # For model parallel runs, siblings share a data coordinate and therefore
    # must also share the same PackedDataset stream.
    from megatron.core import parallel_state as mpu

    try:
        data_parallel_rank = mpu.get_data_parallel_rank(with_context_parallel=False)
        data_parallel_world_size = mpu.get_data_parallel_world_size(with_context_parallel=False)
    except Exception:
        data_parallel_rank = args.rank
        data_parallel_world_size = args.world_size

    rank_seed = get_reference_data_seed(
        getattr(args, 'seed', 4396),
        data_parallel_world_size=data_parallel_world_size,
        data_parallel_rank=data_parallel_rank,
    )
    data_config_kwargs = _build_data_config_kwargs(args, vae_image_downsample)

    def make_dataset():
        data_config = DataConfig(grouped_datasets=copy.deepcopy(dataset_meta), **data_config_kwargs)
        packed_kwargs = _build_packed_dataset_kwargs(
            args,
            data_config=data_config,
            tokenizer=tokenizer,
            special_tokens=special_token_ids,
            data_parallel_rank=data_parallel_rank,
            data_parallel_world_size=data_parallel_world_size,
        )
        dataset = PackedDataset(**packed_kwargs)
        dataset.set_epoch(getattr(args, 'data_seed', 42))
        return dataset, packed_kwargs

    train_dataloader = None
    valid_dataloader = None
    test_dataloader = None

    if train_val_test_num_samples[0] > 0:
        train_dataset, train_kwargs = make_dataset()
        train_dataloader = iter(_build_data_loader(train_dataset, args, rank_seed))
        if args.rank == 0:
            print("Created training dataloader with PackedDataset")
            print(f"  - Rank-local RNG seed: {rank_seed}")
            print(f"  - Expected tokens per batch: {train_kwargs['expected_num_tokens']}")
            print(f"  - Padded tokens per batch: {train_kwargs['max_num_tokens']}")
            print(f"  - Buffer size: {train_kwargs['max_buffer_size']}")

    if train_val_test_num_samples[1] > 0:
        valid_dataset, _ = make_dataset()
        valid_dataloader = iter(_build_data_loader(valid_dataset, args, rank_seed))

    if train_val_test_num_samples[2] > 0:
        test_dataset, _ = make_dataset()
        test_dataloader = iter(_build_data_loader(test_dataset, args, rank_seed))

    return train_dataloader, valid_dataloader, test_dataloader
