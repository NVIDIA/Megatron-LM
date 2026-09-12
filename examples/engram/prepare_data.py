# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Prepare disjoint holdouts and a weight-free, single-epoch FineWeb data configuration."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    """Hash an artifact without loading its contents into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_documents(lengths: np.ndarray, seed: int = 1234) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle document IDs and split at the full-document boundary closest to half the tokens."""
    order = list(range(len(lengths)))
    random.Random(seed).shuffle(order)
    order = np.asarray(order, dtype=np.int64)
    cumulative = np.cumsum(lengths[order], dtype=np.int64)
    split = int(np.argmin(np.abs(2 * cumulative - cumulative[-1]))) + 1
    if not 0 < split < len(order):
        raise ValueError("The source must contain enough documents for two nonempty holdouts")
    return order[:split], order[split:]


def prepare(source_config: Path, output: Path, seed: int) -> dict:
    """Create derived holdouts and manifests; never modify or overwrite source artifacts."""
    from megatron.core.datasets.indexed_dataset import IndexedDataset, IndexedDatasetBuilder
    from megatron.core.datasets.utils import get_blend_from_list

    config = json.loads(source_config.read_text())
    config = {
        split: value.split() if isinstance(value, str) else value for split, value in config.items()
    }
    train_prefixes, _ = get_blend_from_list(config["train"])
    valid_prefixes, _ = get_blend_from_list(config["valid"])
    test_prefixes, _ = get_blend_from_list(config["test"])
    if len(valid_prefixes) != 1 or valid_prefixes != test_prefixes:
        raise ValueError("This preparation expects one shared original validation/test prefix")
    if len(set(Path(prefix).resolve() for prefix in train_prefixes)) != len(train_prefixes):
        raise ValueError("Duplicate training prefixes")
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    output.mkdir(parents=True)

    inventory = []
    for prefix in train_prefixes:
        dataset = IndexedDataset(prefix)
        lengths = dataset.sequence_lengths
        inventory.append(
            {
                "prefix": str(Path(prefix).resolve()),
                "tokens": int(lengths.sum(dtype=np.int64)),
                "documents": len(dataset.document_indices) - 1,
                "complete_samples_4096": (int(lengths.sum(dtype=np.int64)) - 1) // 4096,
                "index_sha256": sha256_file(Path(prefix + ".idx")),
                "bin_bytes": Path(prefix + ".bin").stat().st_size,
            }
        )

    holdout = IndexedDataset(valid_prefixes[0])
    sequence_offsets = np.concatenate(([0], np.cumsum(holdout.sequence_lengths, dtype=np.int64)))
    document_offsets = sequence_offsets[holdout.document_indices]
    valid_ids, test_ids = split_documents(np.diff(document_offsets), seed)
    holdout_manifest = {}
    derived = {"train": train_prefixes}
    for split, document_ids in (("valid", valid_ids), ("test", test_ids)):
        prefix = output / split
        builder = IndexedDatasetBuilder(str(prefix) + ".bin", dtype=holdout.index.dtype)
        for document_id in document_ids:
            begin, end = holdout.document_indices[document_id : document_id + 2]
            pieces = [holdout[index] for index in range(begin, end)]
            tokens = np.concatenate(pieces) if pieces else np.array([], dtype=holdout.index.dtype)
            builder.add_document(tokens, [len(piece) for piece in pieces])
        builder.finalize(str(prefix) + ".idx")
        np.save(output / f"{split}_source_document_ids.npy", document_ids)
        written = IndexedDataset(str(prefix))
        expected_tokens = int(np.diff(document_offsets)[document_ids].sum())
        if int(written.sequence_lengths.sum(dtype=np.int64)) != expected_tokens:
            raise RuntimeError("Derived dataset token count changed")
        derived[split] = [str(prefix.resolve())]
        holdout_manifest[split] = {
            "prefix": str(prefix.resolve()),
            "tokens": expected_tokens,
            "documents": len(document_ids),
            "source_mapping_sha256": sha256_file(output / f"{split}_source_document_ids.npy"),
            "index_sha256": sha256_file(Path(str(prefix) + ".idx")),
            "bin_sha256": sha256_file(Path(str(prefix) + ".bin")),
        }

    joined = np.concatenate([valid_ids, test_ids])
    if not np.array_equal(np.sort(joined), np.arange(len(holdout.document_indices) - 1)):
        raise RuntimeError("Holdout document split is not a disjoint exhaustive partition")
    (output / "per_split.json").write_text(json.dumps(derived, indent=2) + "\n")
    for split in ("valid", "test"):
        # Use the existing full-validation implementation for either independent holdout.
        (output / f"full_{split}.json").write_text(
            json.dumps({"train": None, "valid": derived[split], "test": None}, indent=2) + "\n"
        )
    manifest = {
        "source_config": str(source_config.resolve()),
        "source_config_sha256": sha256_file(source_config),
        "holdout_source_prefix": valid_prefixes[0],
        "holdout_seed": seed,
        "train": inventory,
        "train_tokens": sum(item["tokens"] for item in inventory),
        "complete_train_samples_4096": sum(item["complete_samples_4096"] for item in inventory),
        "holdout": holdout_manifest,
        "per_split_sha256": sha256_file(output / "per_split.json"),
        "train_sampling": "No explicit weights; one epoch per shard; truncate blend to budget",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    """Prepare local indexed data without fetching or retokenizing any content."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_config, args.output, args.seed), indent=2))


if __name__ == "__main__":
    main()
