# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Frozen copies of the runtime dataset-config factories of Megatron-LM commit 438ef0025.

The four ``core_gpt_dataset_config_from_args`` bodies below are verbatim (only the function
names differ) from the entry points at that commit, before the providers started building their
runtime config from ``PretrainConfigContainer.dataset``. ``test_dataset_providers.py`` compares
what every entry point builds now against ``pretrain_gpt.py``'s, and lists where the other three
differed from it. The dataset-class selection of the legacy providers is reproduced at the end of
this module.
"""

import json
from typing import Any, List, Optional, Tuple

from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.training.datasets.fim_dataset import GPTFIMDataset, GPTFIMDatasetConfig
from megatron.training.datasets.sft_dataset import MockSFTDataset, SFTDataset
from megatron.training.datasets.varlen_dataset import MockVarlenDataset, VarlenDataset
from megatron.training.global_vars import get_run_config, get_tokenizer
from megatron.training.utils import get_blend_and_blend_per_split

LEGACY_COMMIT = "438ef0025"


# pretrain_gpt.py, lines 420-488 at 438ef0025
def legacy_gpt_dataset_config(args: Any) -> GPTDatasetConfig:
    """Build the GPT (or FIM) dataset config from parsed CLI args."""
    cfg = get_run_config()
    tokenizer = build_tokenizer(args)

    # Sometimes --data-path is too long, instead we parse it from a file.
    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    sequences_per_dataset = None
    if args.per_dataset_sequences_path is not None:
        with open(args.per_dataset_sequences_path, "r") as f:
            sequences_per_dataset = json.load(f)

    data_args = {
        "random_seed": cfg.rng.seed,
        "sequence_length": args.seq_length,
        "blend": blend,
        "blend_per_split": blend_per_split,
        "split": args.split,
        "multiple_validation_sets": args.multiple_validation_sets,
        "full_validation": args.full_validation,
        "num_dataset_builder_threads": args.num_dataset_builder_threads,
        "path_to_cache": args.data_cache_path,
        "mmap_bin_files": args.mmap_bin_files,
        "tokenizer": tokenizer,
        "reset_position_ids": args.reset_position_ids,
        "reset_attention_mask": args.reset_attention_mask,
        "eod_mask_loss": args.eod_mask_loss,
        "create_attention_mask": args.create_attention_mask_in_dataloader,
        "object_storage_cache_path": args.object_storage_cache_path,
        "mid_level_dataset_surplus": args.mid_level_dataset_surplus,
        "allow_ambiguous_pad_tokens": args.allow_ambiguous_pad_tokens,
        "fast_cache_load": args.dataloader_fast_cache_load,
        "sequences_per_dataset": sequences_per_dataset,
        "defer_npy_index_mmap": args.dataloader_defer_npy_index_mmap,
        "context_parallel_size": args.context_parallel_size,
        "data_parallel_size": args.data_parallel_size,
        "sequence_parallel_size": args.tensor_model_parallel_size * args.sequence_parallel,
        "hybrid_context_parallel": args.hybrid_context_parallel,
        "inter_document_masking": args.dataloader_inter_document_masking,
        "sft_mock_dataset_config_json": args.sft_mock_dataset_config_json,
        "varlen_mock_dataset_config_json": args.varlen_mock_dataset_config_json,
        "varlen_sbhd_validation": args.varlen_sbhd_validation,
    }

    # add FIM args to the config
    if args.fim_data:
        extra_tokens = {
            "prefix": args.fim_prefix_token,
            "middle": args.fim_middle_token,
            "suffix": args.fim_suffix_token,
            "pad": args.fim_pad_token,
            "eod": args.fim_eod_token,
        }
        data_args.update(
            {
                "fim_rate": args.fim_rate,
                "fim_spm_rate": args.fim_spm_rate,
                "fim_extra_tokens": extra_tokens,
                "fim_split_sample": args.fim_split_sample,
                "fim_fragment_rate": args.fim_fragment_rate,
                "fim_no_prefix": args.fim_no_prefix,
            }
        )
        return GPTFIMDatasetConfig(**data_args)

    return GPTDatasetConfig(**data_args)


# pretrain_hybrid.py, lines 383-427 at 438ef0025
def legacy_hybrid_dataset_config(args: Any) -> GPTDatasetConfig:
    """Build the GPT dataset config from parsed CLI args."""
    cfg = get_run_config()
    tokenizer = build_tokenizer(args)

    # Sometimes --data-path is too long, instead we parse it from a file.
    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    sequences_per_dataset = None
    if args.per_dataset_sequences_path is not None:
        with open(args.per_dataset_sequences_path, "r") as f:
            sequences_per_dataset = json.load(f)

    return GPTDatasetConfig(
        random_seed=cfg.rng.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.split,
        multiple_validation_sets=args.multiple_validation_sets,
        full_validation=args.full_validation,
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
        data_parallel_size=args.data_parallel_size,
        sequence_parallel_size=args.tensor_model_parallel_size * args.sequence_parallel,
        hybrid_context_parallel=args.hybrid_context_parallel,
        inter_document_masking=args.dataloader_inter_document_masking,
        varlen_mock_dataset_config_json=args.varlen_mock_dataset_config_json,
        varlen_sbhd_validation=args.varlen_sbhd_validation,
    )


# megatron/elastification/pretrain_hybrid_flex.py, lines 464-490 at 438ef0025
def legacy_hybrid_flex_dataset_config(args):
    cfg = get_run_config()
    tokenizer = get_tokenizer()

    # Sometimes --data-path is too long, instead we parse it from a file.
    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    return GPTDatasetConfig(
        random_seed=cfg.rng.seed,
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
        inter_document_masking=getattr(args, 'dataloader_inter_document_masking', False),
    )


# tools/prepare_cache.py, lines 120-160 at 438ef0025
def legacy_prepare_cache_dataset_config(args: Any) -> GPTDatasetConfig:
    """Build the explicit GPTDatasetConfig used for offline cache preparation."""

    tokenizer = build_tokenizer(args)

    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
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
        multiple_validation_sets=args.multiple_validation_sets,
        full_validation=args.full_validation,
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
        data_parallel_size=args.data_parallel_size,
        sequence_parallel_size=args.tensor_model_parallel_size * args.sequence_parallel,
        hybrid_context_parallel=args.hybrid_context_parallel,
    )


# Dataset-class selection of the legacy providers (train_valid_test_datasets_provider), as
# (dataset class, is_packed_sequence) for pretrain_gpt.py and pretrain_hybrid.py, and the dataset
# class alone for pretrain_hybrid_flex.py, whose rank predicate has no packed-sequence parameter.


def legacy_gpt_dataset_class(args):
    """pretrain_gpt.py, lines 500-520 at 438ef0025."""
    if args.sft:
        return (MockSFTDataset if args.mock_data else SFTDataset), True
    if args.use_varlen_dataset:
        dataset_type = MockVarlenDataset if args.mock_data else VarlenDataset
        return dataset_type, not args.varlen_sbhd_validation
    if args.mock_data:
        return MockGPTDataset, False
    if args.fim_data:
        return GPTFIMDataset, False
    return GPTDataset, False


def legacy_hybrid_dataset_class(args):
    """pretrain_hybrid.py, lines 439-457 at 438ef0025."""
    if args.sft:
        return SFTDataset, True
    if args.use_varlen_dataset:
        dataset_type = MockVarlenDataset if args.mock_data else VarlenDataset
        return dataset_type, not args.varlen_sbhd_validation
    return (MockGPTDataset if args.mock_data else GPTDataset), False


def legacy_hybrid_flex_dataset_class(args):
    """megatron/elastification/pretrain_hybrid_flex.py, lines 503-509 at 438ef0025."""
    if args.sft:
        return SFTDataset
    return MockGPTDataset if args.mock_data else GPTDataset
