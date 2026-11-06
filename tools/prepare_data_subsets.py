# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Prepare nested token-budget subsets of local, text-only IndexedDatasets.

Selection operates on indexed sequences, which need not coincide with documents.
No tokens are copied and no content deduplication is performed. Relative source
prefixes retain Megatron's working-directory semantics.
"""

import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.datasets.utils import get_blend_from_list


def _select_source(prefix: str, weight: Fraction, budgets: list[int], seed: int) -> dict:
    """Read native index lengths and calibrate one shared permutation per source."""
    dataset = IndexedDataset(prefix, mmap=False)
    lengths = dataset.sequence_lengths
    count = len(lengths)
    if count == 0 or count > np.iinfo(np.int32).max or np.any(lengths < 0):
        raise ValueError(f"Invalid or unsupported indexed sequence count/lengths: {prefix}")
    dtype = np.dtype(dataset.index.dtype)
    if not np.issubdtype(dtype, np.integer):
        raise ValueError(f"Expected integer token storage: {prefix}")
    index_path, binary_path = Path(prefix + ".idx"), Path(prefix + ".bin")
    index_stat = index_path.stat()
    # Native text indexes have a 34-byte header, lengths/pointers and document boundaries.
    text_index_bytes = 34 + count * 12 + dataset.document_indices.size * 8
    if index_stat.st_size != text_index_bytes:
        raise ValueError(f"Multimodal or unexpected index layout is unsupported: {prefix}")
    tokens = int(np.sum(lengths, dtype=np.int64))
    if tokens <= 0:
        raise ValueError(f"Source contains no tokens: {prefix}")
    binary_stat = binary_path.stat()
    if binary_stat.st_size != tokens * dtype.itemsize:
        raise ValueError(f"Token binary size disagrees with native sequence lengths: {prefix}")
    quotas = [math.ceil(weight * budget) for budget in budgets]
    if quotas[-1] > tokens:
        raise ValueError(f"Requested token quota exceeds source capacity: {prefix}")

    order = np.random.RandomState(seed).permutation(np.arange(count, dtype=np.int32))
    selections = []
    accumulated = 0
    next_budget = 0
    # Bound the temporary gathered-length and cumulative-sum arrays for large indexes.
    for start in range(0, count, 1_000_000):
        cumulative = np.cumsum(lengths[order[start : start + 1_000_000]], dtype=np.int64)
        while next_budget < len(budgets) and quotas[next_budget] <= accumulated + cumulative[-1]:
            offset = int(np.searchsorted(cumulative, quotas[next_budget] - accumulated))
            retained = start + offset + 1
            probability = (retained - 0.5) / count
            if not 0 < probability <= 1 or math.ceil(probability * count) != retained:
                raise ValueError("Cannot encode retained count as a runtime sampling probability")
            selections.append(
                {
                    "requested_tokens": budgets[next_budget],
                    "token_quota": quotas[next_budget],
                    "selected_sequences": retained,
                    "selected_tokens": accumulated + int(cumulative[offset]),
                    "probability": repr(probability),
                }
            )
            next_budget += 1
        if next_budget == len(budgets):
            break
        accumulated += int(cumulative[-1])
    if len(selections) != len(budgets):
        raise ValueError(f"Source lengths did not reach every token quota: {prefix}")
    for path, before in [(index_path, index_stat), (binary_path, binary_stat)]:
        after = path.stat()
        if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
            raise ValueError(f"Source changed during calibration: {prefix}")
    return {
        "prefix": prefix,
        "token_dtype": dtype.name,
        "available_sequences": count,
        "available_tokens": tokens,
        "index_bytes": index_stat.st_size,
        "index_mtime_ns": index_stat.st_mtime_ns,
        "subsets": selections,
    }


def prepare_subsets(
    input_blend: Path, output_dir: Path, token_budgets: list[int], seed: int = 1234
) -> dict:
    """Write training-only blends for nested subsets, refusing existing output directories.

    Args:
        input_blend: JSON containing training weight/prefix pairs and null valid/test entries.
        output_dir: New directory for subset JSONs and summary.json.
        token_budgets: Distinct positive token budgets; sequences are never truncated.
        seed: Shared uint32 seed for the runtime-compatible per-source permutations.

    Returns:
        Summary of actual token/sequence counts, source metadata and generated filenames.
    """
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Output directory must be new: {output_dir}")
    if not token_budgets or any(type(value) is not int or value <= 0 for value in token_budgets):
        raise ValueError("Token budgets must be positive integers")
    if len(set(token_budgets)) != len(token_budgets):
        raise ValueError("Token budgets must be distinct")
    if type(seed) is not int or not 0 <= seed <= np.iinfo(np.uint32).max:
        raise ValueError("Seed must be a uint32 integer")
    budgets = sorted(token_budgets)
    original = input_blend.read_bytes()
    blend = json.loads(original)
    if not isinstance(blend, dict):
        raise ValueError("Input blend must be a JSON object")
    train = blend.get("train")
    if (
        not isinstance(train, list)
        or not train
        or len(train) % 2
        or blend.get("valid") is not None
        or blend.get("test") is not None
    ):
        raise ValueError("Expected training weight/prefix pairs and null valid/test entries")
    weights, seen = [], set()
    for raw_weight, prefix in zip(train[::2], train[1::2]):
        try:
            if isinstance(raw_weight, bool) or not math.isfinite(float(raw_weight)):
                raise ValueError("Weights must be finite and positive")
            weight = Fraction(str(raw_weight))
        except (TypeError, ValueError, ZeroDivisionError) as error:
            raise ValueError("Weights must be finite and positive") from error
        if weight <= 0:
            raise ValueError("Weights must be finite and positive")
        if not isinstance(prefix, str) or not prefix or "://" in prefix:
            raise ValueError("Source prefixes must identify local indexed datasets")
        if prefix != prefix.strip():
            raise ValueError("Source prefixes must not have surrounding whitespace")
        try:
            numeric_prefix = math.isfinite(float(prefix))
        except ValueError:
            numeric_prefix = False
        if numeric_prefix:
            raise ValueError(
                "Numeric source prefixes must use an explicit relative path, e.g. ./123"
            )
        resolved = Path(prefix).resolve()
        if resolved in seen:
            raise ValueError(f"Repeated source prefix: {prefix}")
        seen.add(resolved)
        weights.append(weight)
    total_weight = sum(weights)
    sources = [
        _select_source(prefix, weight / total_weight, budgets, seed)
        for prefix, weight in zip(train[1::2], weights)
    ]
    if input_blend.read_bytes() != original:
        raise ValueError("Input blend changed during calibration")
    summary = {
        "seed": seed,
        "selection_unit": "indexed_sequence",
        "numpy_version": np.__version__,
        "input_blend_sha256": hashlib.sha256(original).hexdigest(),
        "sources": sources,
        "subsets": [],
    }
    outputs = {}
    for index, budget in enumerate(budgets):
        triples = []
        for source_index, source in enumerate(sources):
            triples.extend(
                [
                    train[2 * source_index],
                    train[2 * source_index + 1],
                    source["subsets"][index]["probability"],
                ]
            )
        parsed = get_blend_from_list(triples, allow_per_source_downsampling=True)
        if parsed[0] != train[1::2]:
            raise ValueError("Source prefixes changed when parsing the generated runtime blend")
        actual = sum(source["subsets"][index]["selected_tokens"] for source in sources)
        filename = f"subset_{budget}.json"
        outputs[filename] = {
            "train": triples,
            "valid": None,
            "test": None,
            "downsampling": {
                "seed": seed,
                "selection_unit": "indexed_sequence",
                "requested_tokens": budget,
                "selected_tokens": actual,
            },
        }
        summary["subsets"].append(
            {"file": filename, "requested_tokens": budget, "selected_tokens": actual}
        )
    outputs["summary.json"] = summary
    # Create only after all inputs and results pass. exist_ok=False also excludes concurrent writers.
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, value in outputs.items():
        with (output_dir / name).open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return summary


def main() -> None:
    """Parse the standalone pool-preparation command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-blend", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--token-budgets", type=int, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    result = prepare_subsets(args.input_blend, args.output_dir, args.token_budgets, args.seed)
    print(json.dumps(result["subsets"], indent=2))


if __name__ == "__main__":
    main()
