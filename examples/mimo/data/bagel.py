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
import logging

# Import directly from bagel.data to avoid circular import
from bagel.data.dataset_base import (
    PackedDataset,
    DataConfig,
    collate_wrapper,
)
from bagel.data import dataset_info as bagel_dataset_info
from bagel.data.data_utils import add_special_tokens
from megatron.training import get_args
from megatron.training.utils import print_rank_0

logging.basicConfig(level=logging.INFO, force=True)

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
        logging.debug(f"Added {num_new_tokens} special tokens to tokenizer")
        logging.debug(f"Special token IDs: {special_token_ids}")

    # Configure dataset
    # Note: You need to configure your dataset according to your data structure
    # This is a placeholder configuration that should be customized
    dataset_config_file = os.environ.get("DATA_CONFIG_FILE")
    with open(dataset_config_file, "r") as stream:
        dataset_meta = yaml.safe_load(stream)

    # Replace placeholder 'your_data_path' with actual data root from BAGEL_EXAMPLE_PATH.
    # DATASET_INFO paths are like your_data_path/bagel_example/t2i; base = dirname(BAGEL_EXAMPLE_PATH).
    bagel_example_path = os.environ.get("BAGEL_EXAMPLE_PATH", "").strip()
    if bagel_example_path:
        data_root = os.path.dirname(os.path.abspath(bagel_example_path))
        for _group, datasets in bagel_dataset_info.DATASET_INFO.items():
            for _name, meta in datasets.items():
                for key, val in list(meta.items()):
                    if isinstance(val, str) and "your_data_path" in val:
                        meta[key] = val.replace("your_data_path", data_root)

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

    # Common PackedDataset parameters
    packed_dataset_kwargs = {
        'data_config': data_config,
        'tokenizer': tokenizer,
        'special_tokens': special_token_ids,
        'local_rank': args.rank,
        'world_size': args.world_size,
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
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=1,
            collate_fn=collate_wrapper(),
            num_workers=getattr(args, 'num_workers', 1),
            pin_memory=True,
        )

        if args.rank == 0:
            logging.debug(f"Created training dataloader with PackedDataset")
            logging.debug(f"  - Expected tokens per batch: {train_kwargs['expected_num_tokens']}")
            logging.debug(f"  - Max tokens: {train_kwargs['max_num_tokens']}")
            logging.debug(f"  - Buffer size: {train_kwargs['max_buffer_size']}")

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

        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=1,
            collate_fn=collate_wrapper(),
            num_workers=getattr(args, 'num_workers', 1),
            pin_memory=True,
        )

    if train_val_test_num_samples[2] > 0:
        test_dataset = PackedDataset(**packed_dataset_kwargs)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            collate_fn=collate_wrapper(),
            num_workers=getattr(args, 'num_workers', 1),
            pin_memory=True,
        )

    return train_dataloader, valid_dataloader, test_dataloader


def bagel_packed_batch_to_mimo_batch(packed_batch, diffusion_wrapper: DiffusionWrapper=None):
    """
    Convert PackedDataset batch format to MIMO model input format.

    Args:
        packed_batch: SimpleCustomBatch object from PackedDataset

    Returns:
        Dict with MIMO model inputs
    """
    # Convert to dict if it's a SimpleCustomBatch object
    if hasattr(packed_batch, 'to_dict'):
        batch_dict = packed_batch.to_dict()
    else:
        batch_dict = packed_batch

    # Get sequence length (total length including text + vision tokens)
    seq_len = batch_dict['sequence_length']

    # For Bagel, input_ids is just packed_text_ids (text tokens only)
    # The full sequence will be constructed in the model
    input_ids = batch_dict['packed_text_ids'].unsqueeze(0)  # (1, num_text_tokens)

    # Create position_ids for text tokens only (for MIMO's get_text_embeddings)
    # This matches the length of input_ids
    text_seq_len = input_ids.shape[1]
    position_ids = torch.arange(text_seq_len, dtype=torch.long).unsqueeze(0)

    # NOTE: Don't create BlockMask here! It cannot be broadcast via object_list.
    # Instead, pass the raw data (split_lens, attn_modes) and create BlockMask in the model.
    # The attention_mask will be created in hf_bagel_llm.py
    attention_mask = None  # Will be created in the model

    # Prepare labels and loss_mask
    if 'packed_label_ids' in batch_dict:
        labels = torch.full((1, seq_len), fill_value=-100, dtype=torch.long)
        ce_loss_indexes = batch_dict['ce_loss_indexes']
        labels[:, ce_loss_indexes] = batch_dict['packed_label_ids']

        loss_mask = torch.zeros((1, seq_len), dtype=torch.float)
        loss_mask[:, ce_loss_indexes] = torch.tensor(batch_dict['ce_loss_weights'])
    else:
        labels = None
        loss_mask = None

    # Prepare final batch
    mimo_batch = {
        'input_ids': input_ids,
        'position_ids': position_ids,
        'labels': labels,
        'loss_mask': loss_mask,
    }

    # Prepare modality inputs for vision encoder
    modality_inputs = {}
    if 'packed_vit_tokens' in batch_dict:
        modality_inputs['images'] = {
            'vision_encoder': {
                'packed_vit_tokens': batch_dict['packed_vit_tokens'],
                'vit_token_seqlens': batch_dict['vit_token_seqlens'],
                'packed_vit_position_ids': batch_dict['packed_vit_position_ids'],
            }
        }

    if "packed_vae_token_indexes" in batch_dict:
        vis_gen_loss_inputs, vis_gen_modality_inputs = bagel_process_gen_data(batch_dict, diffusion_wrapper)
        modality_inputs['diffusion'] = vis_gen_modality_inputs
        mimo_batch.update(vis_gen_loss_inputs)
    else:
        logging.debug("packed_vae_token_indexes is not in batch_dict, skip visual gen data preparation")


    if modality_inputs:
        mimo_batch['modality_inputs'] = modality_inputs

    # Bagel-specific parameters for packed sequence training
    mimo_batch['sample_lens'] = batch_dict['sample_lens']

    # Full sequence length (text + vision tokens)
    mimo_batch['sequence_length'] = seq_len

    # Position IDs for the FULL sequence (text + vision)
    mimo_batch['packed_position_ids'] = batch_dict['packed_position_ids']

    # Text token indexes in the full sequence
    mimo_batch['packed_text_indexes'] = batch_dict['packed_text_indexes']

    # Vision token indexes in the full sequence
    if 'packed_vit_token_indexes' in batch_dict:
        mimo_batch['packed_vit_token_indexes'] = batch_dict['packed_vit_token_indexes']
    else:
        logger.debug("packed_vit_token_indexes is not in batch_dict")

    # Loss-related
    if 'ce_loss_indexes' in batch_dict:
        mimo_batch['ce_loss_indexes'] = batch_dict['ce_loss_indexes']

    if 'packed_label_ids' in batch_dict:
        mimo_batch['packed_label_ids'] = batch_dict['packed_label_ids']

    # Pass split_lens and attn_modes for creating BlockMask in the model
    if 'split_lens' in batch_dict:
        mimo_batch['split_lens'] = batch_dict['split_lens']
    if 'attn_modes' in batch_dict:
        mimo_batch['attn_modes'] = batch_dict['attn_modes']

    # Pass nested_attention_masks if available (from use_flex=False mode)
    # When nested_attention_masks is provided, it will be used directly as attention_mask
    # Otherwise, BlockMask will be created from split_lens and attn_modes
    if 'nested_attention_masks' in batch_dict:
        mimo_batch['nested_attention_masks'] = batch_dict['nested_attention_masks']

    return mimo_batch

def bagel_process_gen_data(batch_dict: dict, diffusion_wrapper: DiffusionWrapper):
    """
    Process visual generation data for Bagel model.

    Args:
        batch_dict: dict, batch dictionary from PackedDataset
        diffusion_wrapper: DiffusionWrapper, diffusion wrapper for visual generation

    Returns:
        loss_inputs: dict, loss inputs for visual generation
        modality_inputs: dict, modality inputs for visual generation
    """
    assert diffusion_wrapper is not None, "diffusion_wrapper is not provided"

    loss_inputs = {}
    # if visual generation is needed.
    modality_inputs = {}
    if "packed_timesteps" in batch_dict:
        packed_timesteps = batch_dict['packed_timesteps'].cuda()
        shifted_timesteps = diffusion_wrapper.shift_timesteps(packed_timesteps)
        shifted_timesteps.requires_grad = True # for fsdp backward hook
        modality_inputs['shifted_timesteps'] = shifted_timesteps
        loss_inputs['mse_loss_indexes'] = batch_dict['mse_loss_indexes']

    # VAE token indexes in the full sequence
    loss_inputs['packed_vae_token_indexes'] = batch_dict['packed_vae_token_indexes']

    #vae encode
    padded_images = batch_dict['padded_images'].cuda()
    latents = diffusion_wrapper.vae_encode(padded_images, batch_dict['patchified_vae_latent_shapes'])
    #add noise and get target for mse loss
    if 'packed_timesteps' in batch_dict:
        latents, _, target = diffusion_wrapper.add_noise(latents, shifted_timesteps)
        loss_inputs['vis_gen_target'] = target

    modality_inputs.update({
        'latents': latents,
        'latent_position_ids': batch_dict['packed_latent_position_ids'],
    })

    return loss_inputs, modality_inputs
