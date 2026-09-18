# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Compare the model tensors of two Megatron distributed checkpoints (torch_dist format).

Loads both checkpoints as plain (unsharded) tensors on one process and reports, per model
tensor, the maximum absolute difference, sorted by difference. Used to find which parameters an
optimizer step or a reload changed::

    python tools/dsv41/compare_dist_ckpts.py <ckpt_dir/iter_A> <ckpt_dir/iter_B> [--top 30]
"""

import argparse
import re

import torch

from megatron.core.dist_checkpointing import load_plain_tensors

_SKIP_PREFIXES = (
    "optimizer",
    "opt_param_scheduler",
    "rng_state",
    "rerun_state_machine",
    "iteration",
    "args",
    "checkpoint_version",
    "num_floating_point",
)


def _model_tensors(state):
    out = {}
    for key, value in state.items():
        if key.startswith(_SKIP_PREFIXES) or not isinstance(value, torch.Tensor):
            continue
        out[key] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--list-identical", action="store_true")
    args = parser.parse_args()
    identical_keys = []
    if not torch.distributed.is_initialized():
        import os

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29581")
        torch.distributed.init_process_group("gloo", rank=0, world_size=1)

    raw_a, raw_b = load_plain_tensors(args.a), load_plain_tensors(args.b)
    print(f"A: {len(raw_a)} keys, e.g. {sorted(raw_a)[:3]}")
    a, b = _model_tensors(raw_a), _model_tensors(raw_b)
    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    if only_a or only_b:
        print(f"keys only in A: {only_a[:10]} (+{max(0, len(only_a) - 10)})")
        print(f"keys only in B: {only_b[:10]} (+{max(0, len(only_b) - 10)})")
    rows = []
    identical = 0
    for key in sorted(set(a) & set(b)):
        ta, tb = a[key].float(), b[key].float()
        if ta.shape != tb.shape:
            rows.append((float("inf"), key, f"shape {tuple(ta.shape)} vs {tuple(tb.shape)}"))
            continue
        diff = (ta - tb).abs().max().item()
        scale = ta.abs().max().item()
        if diff == 0.0:
            identical += 1
            identical_keys.append(key)
        rows.append(
            (
                diff,
                key,
                f"max|A| {scale:.3e} max|B| {tb.abs().max().item():.3e} "
                f"mean|A| {ta.abs().mean().item():.3e} mean|B| {tb.abs().mean().item():.3e} "
                f"rel {diff / max(scale, 1e-12):.3e}",
            )
        )
    rows.sort(key=lambda r: -r[0])
    print(f"{len(rows)} common tensors, {identical} identical")
    if args.list_identical:
        print("-- identical tensors:")
        for key in identical_keys:
            print(f"  {key}")
    families = {}
    for diff, key, _ in rows:
        fam = re.sub(r"\.\d+\.", ".N.", key)
        fam = re.sub(r"weight\d+$", "weightN", fam)
        families.setdefault(fam, []).append(diff)
    print("-- changed tensor families (max diff, count):")
    for fam, diffs in sorted(families.items(), key=lambda kv: -max(kv[1])):
        if max(diffs) > 0:
            print(f"  {max(diffs):.3e}  x{len(diffs)}  {fam}")
    print(f"-- top {args.top} tensors:")
    for diff, key, note in rows[: args.top]:
        print(f"  {diff:.3e}  {key}  {note}")


if __name__ == "__main__":
    main()
