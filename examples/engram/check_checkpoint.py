# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Inspect native checkpoint completeness and compare a continuous/resumed pair on CPU."""

import argparse
import json
import math
from pathlib import Path


def is_optimizer_key(key: str) -> bool:
    """Recognize native optimizer entries, including nested ChainedOptimizer prefixes."""
    return key.startswith(("optimizer.", "chained_"))


def inspect_checkpoint(path: Path, expected_iteration: int, engram: bool) -> dict:
    """Read native common state and storage metadata, without constructing the model."""
    from torch.distributed.checkpoint import FileSystemReader

    from megatron.core.dist_checkpointing.serialization import load_common_state_dict

    common = load_common_state_dict(path)
    metadata = FileSystemReader(str(path)).read_metadata()
    names = sorted(metadata.state_dict_metadata)
    args = common.get("args")
    schedule = common.get("opt_param_scheduler", {})
    iteration = common.get("iteration")
    samples = getattr(args, "consumed_train_samples", None)
    rng_keys = [key for key in names if key.startswith("rng_state/")]
    sparse_keys = [key for key in names if "optimizer.engram_sparse." in key]
    errors = []
    if iteration != expected_iteration:
        errors.append(f"Checkpoint iteration is {iteration}, expected {expected_iteration}")
    if samples != expected_iteration * 64:
        errors.append(f"Consumed training samples is {samples}, expected {expected_iteration * 64}")
    if schedule.get("num_steps") != samples:
        errors.append("Scheduler sample progress disagrees with consumed training samples")
    if not common.get("optimizer"):
        errors.append("Optimizer common state is absent")
    if not rng_keys:
        errors.append("RNG state shards are absent")
    if engram and not sparse_keys:
        errors.append("Engram sparse optimizer buckets/parameter groups are absent")
    return {
        "passed": not errors,
        "errors": errors,
        "checkpoint": str(path),
        "iteration": iteration,
        "consumed_train_samples": samples,
        "consumed_valid_samples": getattr(args, "consumed_valid_samples", None),
        "seed": getattr(args, "seed", None),
        "scheduler": schedule,
        "rng_shards": len(rng_keys),
        "sparse_optimizer_objects": len(sparse_keys),
        "optimizer_storage_entries": len([key for key in names if is_optimizer_key(key)]),
        "storage_entries": len(names),
        "scope": "Completeness/progress inspection; tensor values require compare/resume tests",
    }


def compare_model_tensors(left: Path, right: Path) -> dict:
    """Compare native logical model tensors one at a time, including persistent hash buffers.

    CPU memory is bounded by the largest logical tensor, rather than both full
    models plus all optimizer moments. Optimizer state equality is a separate gate.
    """
    import torch
    from torch.distributed.checkpoint import FileSystemReader, load
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    readers = [FileSystemReader(str(path)) for path in (left, right)]
    metadata = [reader.read_metadata().state_dict_metadata for reader in readers]
    keys = [
        {
            key
            for key, value in entries.items()
            if isinstance(value, TensorStorageMetadata) and not is_optimizer_key(key)
        }
        for entries in metadata
    ]
    if keys[0] != keys[1]:
        return {
            "passed": False,
            "left_only": sorted(keys[0] - keys[1]),
            "right_only": sorted(keys[1] - keys[0]),
        }
    differences = []
    numel = 0
    for key in sorted(keys[0]):
        left_meta, right_meta = metadata[0][key], metadata[1][key]
        if (
            left_meta.size != right_meta.size
            or left_meta.properties.dtype != right_meta.properties.dtype
        ):
            differences.append({"key": key, "reason": "shape/dtype differs"})
            continue
        values = []
        for reader in readers:
            state = {
                key: torch.empty(left_meta.size, dtype=left_meta.properties.dtype, device="cpu")
            }
            load(state, storage_reader=reader, no_dist=True)
            values.append(state[key])
        numel += math.prod(left_meta.size)
        if not torch.equal(*values):
            different, maximum = 0, 0.0
            for begin in range(0, values[0].numel(), 1000000):
                chunks = [value.reshape(-1)[begin : begin + 1000000] for value in values]
                different += int(torch.count_nonzero(chunks[0] != chunks[1]))
                maximum = max(maximum, float((chunks[0].double() - chunks[1].double()).abs().max()))
            differences.append({"key": key, "different_elements": different, "max_abs": maximum})
    return {
        "passed": not differences,
        "tensor_keys": len(keys[0]),
        "elements_compared": numel,
        "exact_equal": not differences,
        "differences": differences,
        "scope": "Logical model tensors/buffers only; excludes optimizer objects and RNG",
    }


def main() -> None:
    """Write explicit checkpoint evidence without modifying either input checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--engram", action="store_true")
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inspect_checkpoint(args.checkpoint, args.iteration, args.engram)
    if args.compare:
        result["model_comparison"] = compare_model_tensors(args.checkpoint, args.compare)
        result["passed"] = result["passed"] and result["model_comparison"]["passed"]
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
