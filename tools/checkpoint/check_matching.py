#!/usr/bin/env python3

import argparse
import sys
import os
from typing import Any, Dict, List, Tuple

import torch


# Ensure repository root is importable so pickled references like 'megatron.core' resolve
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare tensors under 'model' from two torch checkpoints and verify similarity."
        )
    )
    parser.add_argument(
        "file_a",
        type=str,
        help="Path to first checkpoint file (e.g., model_optim_rng.pt)",
    )
    parser.add_argument(
        "file_b",
        type=str,
        help="Path to second checkpoint file (e.g., model_optim_rng.pt)",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-4,
        help="Relative tolerance for floating comparisons (default: 1e-4)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for floating comparisons (default: 1e-6)",
    )
    parser.add_argument(
        "--strict-dtype",
        action="store_true",
        help="Require identical dtypes for compared tensors (default: off)",
    )
    parser.add_argument(
        "--max-report",
        type=int,
        default=50,
        help="Maximum number of mismatches to print (default: 50)",
    )
    return parser.parse_args()


def load_model_container(checkpoint_path: str) -> Any:
    obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "model" not in obj:
        raise KeyError(
            f"Checkpoint at {checkpoint_path} does not contain a top-level 'model' key"
        )
    return obj["model"]


def _flatten_into(prefix: str, obj: Any, out: Dict[str, torch.Tensor]) -> None:
    if torch.is_tensor(obj):
        key = prefix[:-1] if prefix.endswith(".") else prefix
        out[key] = obj.detach().cpu()
        return
    if isinstance(obj, dict):
        for child_key, child_val in obj.items():
            _flatten_into(f"{prefix}{child_key}.", child_val, out)
        return
    if isinstance(obj, (list, tuple)):
        for index, child_val in enumerate(obj):
            _flatten_into(f"{prefix}{index}.", child_val, out)
        return
    # Other types are ignored (e.g., scalars/strings/None)


def collect_tensor_map(model_container: Any) -> Dict[str, torch.Tensor]:
    flattened: Dict[str, torch.Tensor] = {}
    # If the top-level is a list/tuple of module dicts, add a stable prefix
    if isinstance(model_container, (list, tuple)):
        for module_index, module_obj in enumerate(model_container):
            _flatten_into(f"model[{module_index}].", module_obj, flattened)
    else:
        _flatten_into("", model_container, flattened)
    return flattened


def to_comparable_tensor(t: torch.Tensor) -> torch.Tensor:
    if t.is_floating_point():
        # Upcast for numeric stability in comparisons
        return t.to(dtype=torch.float32)
    return t


def compare_tensor_maps(
    a: Dict[str, torch.Tensor],
    b: Dict[str, torch.Tensor],
    *,
    rtol: float,
    atol: float,
    strict_dtype: bool,
    max_report: int,
) -> Tuple[int, int, int]:
    keys_a = set(a.keys())
    keys_b = set(b.keys())

    missing_in_b = sorted(keys_a - keys_b)
    missing_in_a = sorted(keys_b - keys_a)

    mismatch_count = 0

    if missing_in_b:
        print(f"Keys present only in A (missing in B): {len(missing_in_b)}")
        for key in missing_in_b[:max_report]:
            print(f"  - {key}")
        if len(missing_in_b) > max_report:
            print(f"  ... and {len(missing_in_b) - max_report} more")

    if missing_in_a:
        print(f"Keys present only in B (missing in A): {len(missing_in_a)}")
        for key in missing_in_a[:max_report]:
            print(f"  - {key}")
        if len(missing_in_a) > max_report:
            print(f"  ... and {len(missing_in_a) - max_report} more")

    shared_keys = sorted(keys_a & keys_b)

    printed = 0
    for key in shared_keys:
        ta = a[key]
        tb = b[key]

        if strict_dtype and ta.dtype != tb.dtype:
            if printed < max_report:
                print(
                    f"Mismatch dtype for {key}: {ta.dtype} (A) vs {tb.dtype} (B)"
                )
            mismatch_count += 1
            printed += 1
            continue

        if ta.shape != tb.shape:
            if printed < max_report:
                print(
                    f"Mismatch shape for {key}: {tuple(ta.shape)} (A) vs {tuple(tb.shape)} (B)"
                )
            mismatch_count += 1
            printed += 1
            continue

        ca = to_comparable_tensor(ta)
        cb = to_comparable_tensor(tb)

        # Align dtypes for comparison when not strict on dtype
        if ca.dtype != cb.dtype:
            try:
                cb = cb.to(dtype=ca.dtype)
            except Exception:
                # If conversion fails, count as mismatch
                if printed < max_report:
                    print(
                        f"Cannot cast dtype for comparison for {key}: {ca.dtype} (A) vs {cb.dtype} (B)"
                    )
                mismatch_count += 1
                printed += 1
                continue

        equal: bool
        if ca.is_floating_point() and cb.is_floating_point():
            equal = torch.allclose(ca, cb, rtol=rtol, atol=atol)
        else:
            equal = torch.equal(ca, cb)

        if not equal:
            mismatch_count += 1
            if printed < max_report:
                try:
                    abs_diff = (ca - cb).abs()
                    max_abs = abs_diff.max().item() if abs_diff.numel() > 0 else 0.0
                    mean_abs = abs_diff.mean().item() if abs_diff.numel() > 0 else 0.0
                    denom = cb.abs() + 1e-12
                    rel_diff = (abs_diff / denom)
                    max_rel = rel_diff.max().item() if rel_diff.numel() > 0 else 0.0
                    print(
                        f"Value mismatch for {key}: max_abs={max_abs:.6g}, mean_abs={mean_abs:.6g}, max_rel={max_rel:.6g}"
                    )
                except Exception as e:
                    print(f"Value mismatch for {key} (could not compute stats: {e})")
            printed += 1

    if mismatch_count == 0 and not missing_in_a and not missing_in_b:
        print("All tensors under 'model' are similar within the specified tolerances.")
    else:
        print(
            f"Summary: {mismatch_count} value mismatches, {len(missing_in_b)} keys only in A, {len(missing_in_a)} keys only in B."
        )

    return mismatch_count, len(missing_in_b), len(missing_in_a)


def main() -> int:
    args = parse_args()

    try:
        model_a = load_model_container(args.file_a)
        model_b = load_model_container(args.file_b)
    except Exception as e:
        print(f"Failed to load checkpoints: {e}")
        return 2

    map_a = collect_tensor_map(model_a)
    map_b = collect_tensor_map(model_b)

    print(
        f"Loaded {len(map_a)} tensors from A and {len(map_b)} tensors from B under 'model'."
    )
    mismatches, only_in_a, only_in_b = compare_tensor_maps(
        map_a,
        map_b,
        rtol=args.rtol,
        atol=args.atol,
        strict_dtype=args.strict_dtype,
        max_report=args.max_report,
    )

    return 0 if (mismatches == 0 and only_in_a == 0 and only_in_b == 0) else 1


if __name__ == "__main__":
    sys.exit(main())


