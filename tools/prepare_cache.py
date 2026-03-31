# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Prepare GPT dataset caches ahead of training."""

import argparse
import json
from typing import Any, Dict, List, Optional, Tuple

from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
from megatron.core.datasets.utils import compile_helpers
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.training import get_train_valid_test_num_samples
from megatron.training.arguments import parse_args, validate_args
from megatron.training.global_vars import set_args, unset_global_variables
from megatron.training.utils import get_blend_and_blend_per_split

try:
    from megatron.post_training.arguments import add_modelopt_args

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False


def add_prepare_cache_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add cache-preparation specific arguments."""

    group = parser.add_argument_group(title="prepare cache")
    group.add_argument(
        "--prepare-cache-world-size",
        type=int,
        default=None,
        help=(
            "Optional override for the effective world size used to derive data-parallel size and "
            "dataset sample counts during cache preparation."
        ),
    )
    return parser


def _extra_args_provider(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = add_prepare_cache_args(parser)
    if has_nvidia_modelopt:
        parser = add_modelopt_args(parser)
    return parser


def _normalize_prepare_cache_args(args: Any) -> None:
    """Apply cache-preparation specific argument normalization."""

    args.rank = 0

    if args.prepare_cache_world_size is not None:
        if args.prepare_cache_world_size <= 0:
            raise ValueError("--prepare-cache-world-size must be positive")
        args.world_size = args.prepare_cache_world_size


def _validate_prepare_cache_args(args: Any) -> None:
    """Validate options that are intentionally unsupported for offline cache prep."""

    if args.data_cache_path is None:
        raise ValueError("--data-cache-path must be provided for cache preparation")
    if args.mock_data:
        raise ValueError("--mock-data is not supported by tools/prepare_cache.py")
    if getattr(args, "sft", False):
        raise ValueError("--sft is not supported by tools/prepare_cache.py")
    if getattr(args, "fim_data", False):
        raise ValueError("--fim-data is not supported by tools/prepare_cache.py")


def _disable_cache_load_only_flags(args: Any) -> Dict[str, bool]:
    """Disable flags that only make sense when consuming an existing cache."""

    ignored = {
        "dataloader_fast_cache_load": bool(args.dataloader_fast_cache_load),
        "dataloader_defer_npy_index_mmap": bool(args.dataloader_defer_npy_index_mmap),
    }
    args.dataloader_fast_cache_load = False
    args.dataloader_defer_npy_index_mmap = False
    return ignored


def _get_dataset_length(dataset: Optional[Any]) -> Optional[Any]:
    if dataset is None:
        return None
    if isinstance(dataset, list):
        return [len(ds) if ds is not None else None for ds in dataset]
    return len(dataset)


def _print_effective_configuration(
    args: Any, train_valid_test_num_samples: Any, ignored_flags: Dict[str, bool]
) -> None:
    print("> preparing dataset cache with the following effective values:")
    print(f"  world size:         {args.world_size}")
    print(f"  data parallel size: {args.data_parallel_size}")
    print(f"  global batch size:  {args.global_batch_size}")
    print(f"  cache path:         {args.data_cache_path}")
    print(" > datasets target sizes (minimum size):")
    print(f"    train:      {train_valid_test_num_samples[0]}")
    print(f"    validation: {train_valid_test_num_samples[1]}")
    print(f"    test:       {train_valid_test_num_samples[2]}")
    if ignored_flags["dataloader_fast_cache_load"]:
        print("> ignoring --dataloader-fast-cache-load during cache preparation")
    if ignored_flags["dataloader_defer_npy_index_mmap"]:
        print("> ignoring --dataloader-defer-npy-index-mmap during cache preparation")


def core_gpt_dataset_config_from_args(args: Any) -> GPTDatasetConfig:
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


def build_dataset_caches(args: Any) -> Dict[str, Any]:
    """Build the dataset caches for the plain GPTDataset path."""

    _validate_prepare_cache_args(args)
    ignored_flags = _disable_cache_load_only_flags(args)

    unset_global_variables()
    set_args(args)

    try:
        train_valid_test_num_samples = get_train_valid_test_num_samples()
        _print_effective_configuration(args, train_valid_test_num_samples, ignored_flags)

        compile_helpers()

        config = core_gpt_dataset_config_from_args(args)
        train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
            GPTDataset, train_valid_test_num_samples, lambda: True, config
        ).build()

        print("> finished preparing dataset cache")
        print(f"  train dataset length:      {_get_dataset_length(train_ds)}")
        print(f"  validation dataset length: {_get_dataset_length(valid_ds)}")
        print(f"  test dataset length:       {_get_dataset_length(test_ds)}")

        return {
            "world_size": args.world_size,
            "data_parallel_size": args.data_parallel_size,
            "global_batch_size": args.global_batch_size,
            "train_valid_test_num_samples": tuple(train_valid_test_num_samples),
            "train_dataset_length": _get_dataset_length(train_ds),
            "valid_dataset_length": _get_dataset_length(valid_ds),
            "test_dataset_length": _get_dataset_length(test_ds),
        }
    finally:
        unset_global_variables()


def main() -> Dict[str, Any]:
    args = parse_args(
        extra_args_provider=_extra_args_provider,
        ignore_unknown_args=False,
    )
    _normalize_prepare_cache_args(args)
    validate_args(args, defaults={"tokenizer_type": "GPT2BPETokenizer"})
    return build_dataset_caches(args)


if __name__ == "__main__":
    main()
