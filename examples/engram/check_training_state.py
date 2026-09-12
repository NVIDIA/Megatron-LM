# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compare native model, optimizer, scheduler, RNG and data-progress checkpoint state."""

import argparse
import io
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


class StateComparison:
    """Compare nested checkpoint values without printing model data or arbitrary metadata."""

    def __init__(self, atol: float = 0.0, rtol: float = 0.0) -> None:
        self.atol = atol
        self.rtol = rtol
        self.differences = []
        self.leaves = 0
        self.tensor_elements = 0

    def compare(self, left: Any, right: Any, path: str) -> None:
        """Visit every tensor, array and scalar; retain numerical differences even if tolerated."""
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            self._tensor(left, right, path)
        elif isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
            self._tensor(torch.from_numpy(left.copy()), torch.from_numpy(right.copy()), path)
        elif isinstance(left, dict) and isinstance(right, dict):
            if left.keys() != right.keys():
                self._different(path, "dictionary keys differ")
            for key in left.keys() & right.keys():
                self.compare(left[key], right[key], f"{path}/{key}")
        elif isinstance(left, (list, tuple)) and type(left) is type(right):
            if len(left) != len(right):
                self._different(path, "sequence lengths differ")
            for index, (first, second) in enumerate(zip(left, right)):
                self.compare(first, second, f"{path}/{index}")
        elif isinstance(left, io.BytesIO) and isinstance(right, io.BytesIO):
            self.compare(left.getvalue(), right.getvalue(), path)
        else:
            self.leaves += 1
            if type(left) is not type(right):
                self._different(path, "value types differ")
            elif isinstance(left, float):
                if not math.isfinite(left) or not math.isfinite(right):
                    self._different(path, "non-finite scalar")
                elif left != right:
                    self.differences.append(
                        {
                            "path": path,
                            "reason": "floating scalar differs",
                            "max_abs": abs(left - right),
                            "within_tolerance": math.isclose(
                                left, right, abs_tol=self.atol, rel_tol=self.rtol
                            ),
                        }
                    )
            elif left != right:
                self._different(path, "non-floating value differs")

    def _different(self, path: str, reason: str) -> None:
        self.differences.append({"path": path, "reason": reason, "within_tolerance": False})

    def _tensor(self, left: torch.Tensor, right: torch.Tensor, path: str) -> None:
        self.leaves += 1
        if left.shape != right.shape or left.dtype != right.dtype:
            self._different(path, "tensor shape/dtype differs")
            return
        self.tensor_elements += left.numel()
        different, failed, max_abs = 0, 0, 0.0
        floating = left.is_floating_point() or left.is_complex()
        for begin in range(0, left.numel(), 1000000):
            first = left.reshape(-1)[begin : begin + 1000000]
            second = right.reshape(-1)[begin : begin + 1000000]
            if torch.equal(first, second) and (not floating or torch.isfinite(first).all()):
                continue
            different += int(torch.count_nonzero(first != second))
            if floating:
                failed += int(
                    torch.count_nonzero(
                        ~torch.isclose(first, second, atol=self.atol, rtol=self.rtol)
                        | ~torch.isfinite(first)
                        | ~torch.isfinite(second)
                    )
                )
                comparison_dtype = torch.complex128 if first.is_complex() else torch.float64
                delta = (first.to(comparison_dtype) - second.to(comparison_dtype)).abs()
                if delta.numel() and torch.isfinite(delta).all():
                    max_abs = max(max_abs, float(delta.max()))
            else:
                failed += int(torch.count_nonzero(first != second))
        if different or failed:
            self.differences.append(
                {
                    "path": path,
                    "reason": "tensor values differ or are non-finite",
                    "different_elements": different,
                    "outside_tolerance_elements": failed,
                    "max_abs": max_abs if floating else None,
                    "within_tolerance": failed == 0,
                }
            )

    def result(self) -> dict:
        """Report exact equality separately from the explicitly requested tolerance result."""
        return {
            "passed": all(item["within_tolerance"] for item in self.differences),
            "exact_equal": not self.differences,
            "leaves_compared": self.leaves,
            "tensor_elements": self.tensor_elements,
            "atol": self.atol,
            "rtol": self.rtol,
            "differences": self.differences,
        }


def category(key: str) -> str:
    """Classify native storage keys, retaining complete sparse object payloads."""
    if "optimizer.engram_sparse." in key:
        return "sparse_optimizer"
    if key.startswith(("optimizer.", "chained_")):
        return "dense_optimizer"
    if key.startswith("rng_state/"):
        return "rng"
    return "model"


def sparse_row_count(value: Any) -> int:
    """Count coordinates inside native ShardedObject payload lists."""
    if isinstance(value, dict):
        coordinates = value.get("coordinates")
        if isinstance(coordinates, torch.Tensor):
            return coordinates.shape[0]
        return sum(sparse_row_count(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return sum(sparse_row_count(child) for child in value)
    return 0


def storage_batches(
    keys: list[str], entries: dict, object_sizes: dict, limit: int = 64 * 1024**2
) -> list[list[str]]:
    """Bound each side's load buffer, except when one logical tensor alone exceeds the limit."""
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    result, batch, size = [], [], 0
    for key in keys:
        descriptor = entries[key]
        estimate = (
            math.prod(descriptor.size)
            * torch.empty((), dtype=descriptor.properties.dtype).element_size()
            if isinstance(descriptor, TensorStorageMetadata)
            else object_sizes[key]
        )
        if batch and size + estimate > limit:
            result.append(batch)
            batch, size = [], 0
        batch.append(key)
        size += estimate
    if batch:
        result.append(batch)
    return result


def compare_checkpoints(
    left: Path, right: Path, atol: float = 0.0, rtol: float = 0.0, ignore_rng: bool = False
) -> dict:
    """Read one global tensor/object at a time and compare all native training state.

    With topology changes, upstream intentionally does not restore incompatible
    per-rank RNG streams. ``ignore_rng`` records that exclusion explicitly; it must
    not be used to claim exact same-topology continuation.
    """
    from torch.distributed.checkpoint import FileSystemReader, load
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    from megatron.core.dist_checkpointing.serialization import load_common_state_dict

    class CachedReader(FileSystemReader):
        """Keep immutable storage metadata while DCP repeatedly plans bounded loads."""

        def read_metadata(self) -> Any:
            if not hasattr(self, "cached_metadata"):
                self.cached_metadata = super().read_metadata()
            self.storage_data = self.cached_metadata.storage_data
            return self.cached_metadata

    readers = [CachedReader(str(path)) for path in (left, right)]
    metadata = [reader.read_metadata().state_dict_metadata for reader in readers]
    common = [load_common_state_dict(path) for path in (left, right)]
    comparison = StateComparison(atol, rtol)
    for field in ("iteration", "optimizer", "opt_param_scheduler"):
        if any(field not in state for state in common):
            comparison._different(f"common/{field}", "required common state is absent")
        else:
            comparison.compare(common[0][field], common[1][field], f"common/{field}")
    progress = ("consumed_train_samples", "consumed_valid_samples", "seed")
    for field in progress:
        values = [getattr(state.get("args"), field, None) for state in common]
        if field == "consumed_train_samples" and any(value is None for value in values):
            comparison._different(f"progress/{field}", "required sample progress is absent")
        comparison.compare(*values, f"progress/{field}")
    keys = []
    skipped = []
    for entries in metadata:
        selected = set()
        for key in entries:
            if key == "common_state" or key.startswith("common_state/"):
                continue
            if ignore_rng and category(key) == "rng":
                skipped.append(key)
                continue
            selected.add(key)
        keys.append(selected)
    if keys[0] != keys[1]:
        comparison._different("storage", "global storage key sets differ")
    counts = defaultdict(int)
    sparse_rows = 0
    object_sizes = defaultdict(int)
    for index, storage in readers[0].cached_metadata.storage_data.items():
        object_sizes[index.fqn] += storage.length
    batches = storage_batches(sorted(keys[0] & keys[1]), metadata[0], object_sizes)
    for batch in batches:
        states = []
        for reader, entries in zip(readers, metadata):
            state = {}
            for key in batch:
                descriptor = entries[key]
                state[key] = (
                    torch.empty(descriptor.size, dtype=descriptor.properties.dtype)
                    if isinstance(descriptor, TensorStorageMetadata)
                    else None
                )
            load(state, storage_reader=reader, no_dist=True)
            states.append(state)
        for key in batch:
            values = [state.pop(key) for state in states]
            counts[category(key)] += 1
            if category(key) == "sparse_optimizer":
                sparse_rows += sparse_row_count(values[0])
            comparison.compare(*values, key)
        del states
    if not counts["dense_optimizer"]:
        comparison._different("dense_optimizer", "no native optimizer storage entries found")
    if not ignore_rng and not counts["rng"]:
        comparison._different("rng", "no RNG storage entries found")
    result = comparison.result()
    result.update(
        {
            "left": str(left),
            "right": str(right),
            "storage_entries_compared": dict(counts),
            "sparse_rows_compared": sparse_rows,
            "storage_load_batches": len(batches),
            "left_only_keys": sorted(keys[0] - keys[1]),
            "right_only_keys": sorted(keys[1] - keys[0]),
            "rng_comparison": "excluded for native topology change" if ignore_rng else "required",
            "excluded_rng_keys": sorted(set(skipped)),
            "scope": "All model/optimizer storage, common optimizer/scheduler, RNG and progress",
        }
    )
    return result


def main() -> None:
    """Write a complete, read-only checkpoint comparison report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--ignore-rng-for-topology-change", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare_checkpoints(
        args.left, args.right, args.atol, args.rtol, args.ignore_rng_for_topology_change
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({key: value for key, value in report.items() if key != "differences"}, indent=2)
    )
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
