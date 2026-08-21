# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Prepare GPT dataset caches ahead of training.

Unsupported configurations:
    --mock-data, --sft, --fim-data, --step-batch-size-schedule
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
from megatron.core.datasets.utils import compile_helpers
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.training.arguments import parse_args, validate_args
from megatron.training.global_vars import set_args, unset_global_variables
from megatron.training.training import update_train_iters
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
    group.add_argument(
        "--prepare-cache-start-iteration",
        type=int,
        default=None,
        help=(
            "Optional current training iteration used to select the phase when "
            "--phase-transition-iterations is set."
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
    args.iteration = getattr(args, "iteration", 0)

    if args.seq_length is None and getattr(args, "encoder_seq_length", None) is None:
        raise ValueError("--seq-length must be provided for cache preparation")

    if args.prepare_cache_world_size is not None:
        if args.prepare_cache_world_size <= 0:
            raise ValueError("--prepare-cache-world-size must be positive")
        args.world_size = args.prepare_cache_world_size

    prepare_cache_start_iteration = getattr(args, "prepare_cache_start_iteration", None)
    if prepare_cache_start_iteration is not None:
        if prepare_cache_start_iteration < 0:
            raise ValueError("--prepare-cache-start-iteration must be non-negative")
        args.iteration = prepare_cache_start_iteration

    # Offline cache construction does not use model execution parameters, but the shared
    # training argument validator requires them. Fill minimal values when omitted.
    if args.micro_batch_size is None:
        args.micro_batch_size = 1
    if args.num_layers is None and args.encoder_num_layers is None:
        args.num_layers = 1
    if args.hidden_size is None:
        args.hidden_size = 1
    if args.num_attention_heads is None:
        args.num_attention_heads = 1
    if args.max_position_embeddings is None:
        args.max_position_embeddings = args.seq_length or args.encoder_seq_length


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
    if getattr(args, "step_batch_size_schedule", None) is not None:
        raise ValueError("--step-batch-size-schedule is not supported by tools/prepare_cache.py")


def _disable_cache_load_only_flags(args: Any) -> Dict[str, bool]:
    """Disable flags that only make sense when consuming an existing cache."""

    ignored = {
        "dataloader_fast_cache_load": bool(args.dataloader_fast_cache_load),
        "dataloader_defer_npy_index_mmap": bool(args.dataloader_defer_npy_index_mmap),
    }
    args.dataloader_fast_cache_load = False
    args.dataloader_defer_npy_index_mmap = False
    return ignored


def _get_prepare_cache_num_samples(args: Any) -> Tuple[Any, Any, Any]:
    """Return train/validation/test sample targets for offline cache construction."""

    if args.train_samples:
        train_samples = args.train_samples
    else:
        train_samples = args.train_iters * args.global_batch_size

    eval_global_batch_size = getattr(args, "eval_global_batch_size", args.global_batch_size)
    eval_iters = args.eval_iters or 0

    if args.full_validation:
        eval_samples = None
    else:
        if args.skip_train:
            validation_eval_iters = eval_iters
        elif args.eval_interval is None or eval_iters == 0:
            validation_eval_iters = 0
        else:
            assert args.train_iters is not None
            total_eval_points = args.train_iters // args.eval_interval + 1
            if args.start_eval_at_iter is not None:
                skipped_eval_points = args.start_eval_at_iter // args.eval_interval
                total_eval_points = max(0, total_eval_points - skipped_eval_points)
            validation_eval_iters = total_eval_points * eval_iters
        eval_samples = validation_eval_iters * eval_global_batch_size

    test_samples = eval_iters * eval_global_batch_size

    if args.phase_transition_iterations:
        total_train_samples = (
            args.train_samples if args.train_samples is not None else train_samples
        )
        phase_transition_samples = [
            0,
            *[
                iteration * args.global_batch_size
                for iteration in args.phase_transition_iterations
            ],
            total_train_samples,
        ]
        current_sample = args.iteration * args.global_batch_size
        last_transition_sample = max(
            sample for sample in phase_transition_samples if sample <= current_sample
        )
        next_transition_sample = min(
            sample for sample in phase_transition_samples if sample > current_sample
        )
        train_samples = next_transition_sample - last_transition_sample

    return train_samples, eval_samples, test_samples


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
    )


def build_dataset_caches(args: Any) -> Dict[str, Any]:
    """Build the dataset caches for the plain GPTDataset path."""

    _validate_prepare_cache_args(args)
    ignored_flags = _disable_cache_load_only_flags(args)

    unset_global_variables()
    set_args(args)

    try:
        # Derive train_iters from --train-samples when needed (pretrain() does the same).
        update_train_iters(args)
        train_valid_test_num_samples = _get_prepare_cache_num_samples(args)
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
    args = parse_args(extra_args_provider=_extra_args_provider, ignore_unknown_args=False)
    _normalize_prepare_cache_args(args)
    validate_args(args, defaults={"tokenizer_type": "GPT2BPETokenizer"})
    return build_dataset_caches(args)


if __name__ == "__main__":
    main()
