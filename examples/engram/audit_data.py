# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Verify actual Megatron sample indices before starting the comparison."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def array_digest(array: np.ndarray) -> str:
    """Hash a contiguous index array, including shape and dtype, without copying it in full."""
    digest = hashlib.sha256(f"{array.shape}:{array.dtype}".encode())
    for begin in range(0, len(array), 1000000):
        digest.update(np.ascontiguousarray(array[begin : begin + 1000000]).tobytes())
    return digest.hexdigest()


def audit_training_indices(dataset: Any, requested_samples: int) -> dict:
    """Prove no-repeat main-label coverage from the instantiated dataset indices.

    The configured corpus has nonnegative token IDs and no EOD loss masking.
    Thus every full-length selected label has unit loss mask. Reject other mask
    settings instead of reporting an unverified coverage percentage.
    """
    if not hasattr(dataset, "datasets") or requested_samples > len(dataset):
        raise ValueError("Expected a sufficiently large blended training dataset")
    blend_ids = dataset.dataset_index[:requested_samples]
    sample_ids = dataset.dataset_sample_index[:requested_samples]
    if np.any(blend_ids < 0) or np.any(blend_ids >= len(dataset.datasets)):
        raise ValueError("Out-of-range blend dataset index")
    shards = []
    total_tokens = 0
    for shard_id, child in enumerate(dataset.datasets):
        config = child.config
        pad_id = getattr(child, "_pad_token_id", -1)
        if (
            config.eod_mask_loss
            or (pad_id is not None and pad_id >= 0)
            or not config.add_extra_token_to_sequence
        ):
            raise ValueError(
                "This audit requires unmasked EOD, negative padding and shifted labels"
            )
        if child.num_samples is not None:
            raise ValueError("Each training shard must use num_samples=None for one epoch")
        selected = sample_ids[blend_ids == shard_id]
        if not np.array_equal(selected, np.arange(len(selected))):
            raise ValueError("Blend indices repeat, skip, or wrap shard sample positions")
        documents = child.document_index
        if not np.array_equal(np.sort(documents), np.sort(child.indices)):
            raise ValueError("Document index is not a single permutation of source sequences")
        shuffle = child.shuffle_index
        if not np.array_equal(np.sort(shuffle), np.arange(len(shuffle))):
            raise ValueError("Shuffle index is not a permutation")
        if len(selected) > len(shuffle):
            raise ValueError("Selected more samples than the one-epoch shard contains")
        lengths = child.dataset.sequence_lengths
        offsets = np.concatenate(([0], np.cumsum(lengths[documents], dtype=np.int64)))
        index = child.sample_index
        if np.any(index[:, 0] < 0) or np.any(index[:, 0] >= len(documents)):
            raise ValueError("Sample index document is out of bounds")
        if np.any(index[:, 1] < 0) or np.any(index[:, 1] > lengths[documents[index[:, 0]]]):
            raise ValueError("Sample index offset is out of bounds")
        positions = offsets[index[:, 0]] + index[:, 1]
        if not np.all(np.diff(positions) == config.sequence_length):
            raise ValueError("Training samples are not contiguous, full-length intervals")
        if positions[-1] >= offsets[-1]:
            raise ValueError("The final next-token target extends beyond the source")
        # Distinct fixed-length intervals in a one-epoch sequence permutation have
        # disjoint label positions [start + 1, end + 1). Context overlap is harmless.
        unique_labels = len(selected) * config.sequence_length
        tokens = int(lengths[child.indices].sum(dtype=np.int64))
        total_tokens += tokens
        shards.append(
            {
                "prefix": child.dataset.path_prefix,
                "source_tokens": tokens,
                "selected_samples": len(selected),
                "unique_main_labels": unique_labels,
                "document_index_sha256": array_digest(documents),
                "sample_index_sha256": array_digest(index),
                "shuffle_index_sha256": array_digest(shuffle),
            }
        )
    labels = sum(shard["unique_main_labels"] for shard in shards)
    coverage = labels / total_tokens
    if coverage < 0.9:
        raise ValueError(f"Main-label source coverage is only {coverage:.6%}")
    return {
        "requested_samples": requested_samples,
        "source_tokens": total_tokens,
        "unique_valid_main_labels": labels,
        "coverage": coverage,
        "blend_index_sha256": array_digest(blend_ids),
        "blend_sample_index_sha256": array_digest(sample_ids),
        "mask_proof": "Full shifted-label intervals; no EOD masking; absent or negative padding ID",
        "mtp_auxiliary_targets_counted_as_new_source": False,
        "shards": shards,
    }


def main() -> None:
    """Build and audit the same native one-epoch indices on CPU before GPU acceptance."""
    from megatron.core.datasets.blended_megatron_dataset_builder import (
        BlendedMegatronDatasetBuilder,
    )
    from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
    from megatron.core.datasets.utils import get_blend_from_list
    from megatron.core.tokenizers import MegatronTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = MegatronTokenizer.from_pretrained(
        tokenizer_path=args.tokenizer,
        metadata_path={"library": "huggingface"},
        chat_template=None,
        vocab_file=None,
        merges_file=None,
        additional_special_tokens=[],
        use_fast=True,
        trust_remote_code=False,
        include_special_tokens=True,
        use_gigatoken=False,
    )
    sources = json.loads(args.data_config.read_text())
    config = GPTDatasetConfig(
        random_seed=2026,
        sequence_length=4096,
        blend_per_split=[
            get_blend_from_list(sources[split]) for split in ("train", "valid", "test")
        ],
        path_to_cache=args.cache,
        tokenizer=tokenizer,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
    )
    train, _, _ = BlendedMegatronDatasetBuilder(
        GPTDataset, [36754 * 64, (36754 // 500 + 1) * 2 * 64, 2 * 64], lambda: True, config
    ).build()
    audit = audit_training_indices(train, 36754 * 64)
    args.output.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({key: value for key, value in audit.items() if key != "shards"}, indent=2))


if __name__ == "__main__":
    main()
