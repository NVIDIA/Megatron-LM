# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""
Bagel dataloader provider using PackedDataset from bagel.data.

This module provides a dataloader that uses the PackedDataset for efficient
sequence packing and multimodal data loading for Bagel model training.
"""
import os
import copy
import yaml
import torch
from torch.utils.data import DataLoader
from diffusion.diffusion_wrapper import DiffusionWrapper

# Import directly from bagel.data to avoid circular import
from bagel.data.dataset_base import (
    PackedDataset,
    DataConfig,
    collate_wrapper,
)
from bagel.data.data_utils import add_special_tokens
from megatron.training import get_args
from megatron.training.utils import print_rank_0


def bagel_dataloader_provider(train_val_test_num_samples):
    """
    Provide train/valid/test dataloaders using PackedDataset.

    Args:
        train_val_test_num_samples: Tuple of (train_samples, valid_samples, test_samples)

    Returns:
        Tuple of (train_dataloader, valid_dataloader, test_dataloader)
    """
    args = get_args()

    # Get tokenizer
    from bagel.modeling.qwen2 import Qwen2Tokenizer
    tokenizer = Qwen2Tokenizer.from_pretrained(
        args.tokenizer_model if hasattr(args, 'tokenizer_model') else "Qwen/Qwen2-7B"
    )

    # Add special tokens for multimodal training
    tokenizer, special_token_ids, num_new_tokens = add_special_tokens(tokenizer)

    if args.rank == 0:
        print(f"Added {num_new_tokens} special tokens to tokenizer")
        print(f"Special token IDs: {special_token_ids}")

    # Configure dataset
    # Note: You need to configure your dataset according to your data structure
    # This is a placeholder configuration that should be customized
    dataset_config_file = os.environ.get("DATA_CONFIG_FILE")
    with open(dataset_config_file, "r") as stream:
        dataset_meta = yaml.safe_load(stream)

    if getattr(args, 'diffusion_wrapper', None):
        vae_image_downsample = args.diffusion_wrapper.latent_downsample
    else:
        vae_image_downsample = 16

    print_rank_0("dataset_meta", dataset_meta)

    data_config = DataConfig(
        grouped_datasets=dataset_meta,
        text_cond_dropout_prob=getattr(args, 'text_cond_dropout_prob', 0.1),
        vit_cond_dropout_prob=getattr(args, 'vit_cond_dropout_prob', 0.4),
        vae_cond_dropout_prob=getattr(args, 'vae_cond_dropout_prob', 0.1),
        # vae_image_downsample=getattr(args, 'vae_image_downsample', 16),
        vae_image_downsample=vae_image_downsample,
        max_latent_size=getattr(args, 'max_latent_size', 32),
        vit_patch_size=getattr(args, 'vit_patch_size', 14),
        max_num_patch_per_side=getattr(args, 'max_num_patch_per_side', 70),
    )

    # PP-aware data sharding: shard by data-parallel rank (with CP folded in
    # so CP siblings get the same data, matching shard_data_for_cp's
    # invariant). All ranks within a PP group share a dp coord and therefore
    # the same data shard — required for PP>1 because every PP stage runs
    # bagel_packed_batch_to_mimo_batch independently and the resulting
    # packed_seq_params must agree across stages.
    from megatron.core import parallel_state as _mpu
    try:
        _dp_rank = _mpu.get_data_parallel_rank(with_context_parallel=False)
        _dp_world = _mpu.get_data_parallel_world_size(with_context_parallel=False)
    except Exception:
        _dp_rank = args.rank
        _dp_world = args.world_size

    # Common PackedDataset parameters
    packed_dataset_kwargs = {
        'data_config': data_config,
        'tokenizer': tokenizer,
        'special_tokens': special_token_ids,
        'local_rank': _dp_rank,
        'world_size': _dp_world,
        'num_workers': getattr(args, 'num_workers', 1),
        'expected_num_tokens': getattr(args, 'seq_length', 32768),
        'max_num_tokens_per_sample': getattr(args, 'max_num_tokens_per_sample', 16384),
        'max_num_tokens': getattr(args, 'max_num_tokens', 36864),
        'prefer_buffer_before': getattr(args, 'prefer_buffer_before', 16384),
        'max_buffer_size': getattr(args, 'packing_buffer_size', 50) if hasattr(args, 'packing_buffer_size') and args.packing_buffer_size is not None else 50,
        'interpolate_pos': getattr(args, 'interpolate_pos', False),
        'use_flex': getattr(args, 'use_flex_attention', False),
    }

    # Create dataloaders
    train_dataloader = None
    valid_dataloader = None
    test_dataloader = None

    if train_val_test_num_samples[0] > 0:
        # Create training dataset with fresh config copy
        train_data_config = DataConfig(
            grouped_datasets=copy.deepcopy(dataset_meta),
            text_cond_dropout_prob=getattr(args, 'text_cond_dropout_prob', 0.1),
            vit_cond_dropout_prob=getattr(args, 'vit_cond_dropout_prob', 0.4),
            vae_cond_dropout_prob=getattr(args, 'vae_cond_dropout_prob', 0.1),
            vae_image_downsample=getattr(args, 'vae_image_downsample', 16),
            max_latent_size=getattr(args, 'max_latent_size', 32),
            vit_patch_size=getattr(args, 'vit_patch_size', 14),
            max_num_patch_per_side=getattr(args, 'max_num_patch_per_side', 70),
        )

        train_kwargs = packed_dataset_kwargs.copy()
        train_kwargs['data_config'] = train_data_config
        train_dataset = PackedDataset(**train_kwargs)
        # Create training dataloader
        train_dataloader = iter(DataLoader(
            train_dataset,
            batch_size=1,
            collate_fn=collate_wrapper(),
            num_workers=getattr(args, 'num_workers', 1),
            pin_memory=True,
        ))

        if args.rank == 0:
            print(f"Created training dataloader with PackedDataset")
            print(f"  - Expected tokens per batch: {train_kwargs['expected_num_tokens']}")
            print(f"  - Max tokens: {train_kwargs['max_num_tokens']}")
            print(f"  - Buffer size: {train_kwargs['max_buffer_size']}")

    # Note: For validation and test, you might want to use different configurations
    # or different datasets. This is a simplified implementation.
    if train_val_test_num_samples[1] > 0:
        # Create validation dataset with fresh config copy
        valid_data_config = DataConfig(
            grouped_datasets=copy.deepcopy(dataset_meta),
            text_cond_dropout_prob=getattr(args, 'text_cond_dropout_prob', 0.1),
            vit_cond_dropout_prob=getattr(args, 'vit_cond_dropout_prob', 0.4),
            vae_cond_dropout_prob=getattr(args, 'vae_cond_dropout_prob', 0.1),
            vae_image_downsample=getattr(args, 'vae_image_downsample', 16),
            max_latent_size=getattr(args, 'max_latent_size', 32),
            vit_patch_size=getattr(args, 'vit_patch_size', 14),
            max_num_patch_per_side=getattr(args, 'max_num_patch_per_side', 70),
        )

        valid_kwargs = packed_dataset_kwargs.copy()
        valid_kwargs['data_config'] = valid_data_config
        valid_dataset = PackedDataset(**valid_kwargs)

        valid_dataloader = iter(DataLoader(
            valid_dataset,
            batch_size=1,
            collate_fn=collate_wrapper(),
            num_workers=getattr(args, 'num_workers', 1),
            pin_memory=True,
        ))

    if train_val_test_num_samples[2] > 0:
        test_dataset = PackedDataset(**packed_dataset_kwargs)
        test_dataloader = iter(DataLoader(
            test_dataset,
            batch_size=1,
            collate_fn=collate_wrapper(),
            num_workers=getattr(args, 'num_workers', 1),
            pin_memory=True,
        ))

    return train_dataloader, valid_dataloader, test_dataloader
