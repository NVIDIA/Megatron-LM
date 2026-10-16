# Copyright (c) 2026, Zyphra Technologies Inc. All rights reserved.
"""Inspect expert-affinity capture shards produced by megatron/core/capture.py.

Usage:
    python tools/inspect_expert_capture.py /path/to/captures
    python tools/inspect_expert_capture.py /path/to/captures/iter_00002000_mb0_rank0.pt
    python tools/inspect_expert_capture.py /path/to/captures --coactivation

Prints every key with its dtype/shape, cross-checks the shapes against each
other, and summarises which iterations / micro-batches / ranks are present.
With ``--coactivation`` it also reports the per-layer expert co-activation
counts (how often a pair of experts is selected by the same token), which is
the raw signal the affinity / expert-placement analysis is built on.

Read-only; never writes anything.
"""

import argparse
import glob
import os
import re
import sys
from collections import defaultdict

import torch

SHARD_RE = re.compile(r"iter_(\d+)_mb(\d+)_rank(\d+)\.pt$")


def fmt(v):
    """One-line description of a payload value."""
    if isinstance(v, torch.Tensor):
        return f"Tensor {tuple(v.shape)} {str(v.dtype).replace('torch.', '')}"
    if isinstance(v, (list, tuple)):
        return f"{type(v).__name__}(len={len(v)}) e.g. {list(v)[:8]}"
    return f"{type(v).__name__}: {v}"


def collect(paths):
    """Expand files / directories / globs into a sorted list of shard paths."""
    files = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(glob.glob(os.path.join(p, "iter_*_mb*_rank*.pt")))
        else:
            matches = sorted(glob.glob(p))
            files.extend(matches if matches else [p])
    return sorted(set(files))


def describe(path):
    """Print one shard's payload and check its shapes agree. Returns the payload."""
    print(f"\n=== {path} ===")
    d = torch.load(path, weights_only=False, map_location="cpu")
    for k, v in d.items():
        print(f"  {k:20s} = {fmt(v)}")

    ep = d.get("expert_paths")
    if ep is None:
        print("  [warn] no expert_paths in this shard (MOE_ROUTE_CAPTURE_EXPERTS=0?)")
        return d

    b, s, n_moe, topk = ep.shape
    print(f"\n  b={b} s={s} moe_layers={n_moe} topk={topk}")
    print(f"  expert id range: [{int(ep.min())}, {int(ep.max())}]")

    layers = d.get("moe_layer_indices")
    if layers is not None and len(layers) != n_moe:
        print(f"  [error] moe_layer_indices has {len(layers)} entries but expert_paths has {n_moe}")

    probs = d.get("routing_probs")
    if probs is not None and tuple(probs.shape) != (b, s, n_moe, topk):
        print(f"  [error] routing_probs {tuple(probs.shape)} != expert_paths {(b, s, n_moe, topk)}")

    tid = d.get("token_ids")
    if tid is not None and tuple(tid.shape) != (b, s):
        print(f"  [error] token_ids {tuple(tid.shape)} != (b, s) = {(b, s)}")

    emb = d.get("input_embeddings")
    if emb is not None and tuple(emb.shape[:2]) != (b, s):
        print(f"  [error] input_embeddings {tuple(emb.shape)} is not [b, s, h] = [{b}, {s}, *]")

    # A duplicate expert inside one token's top-k slots would mean the capture
    # is misaligned (top-k always selects distinct experts).
    dupes = int((ep.sort(dim=-1).values.diff(dim=-1) == 0).sum())
    print(
        f"  duplicate experts within a token's top-k: {dupes} {'(OK)' if dupes == 0 else '(BAD)'}"
    )

    print(
        f"  layer {layers[-1] if layers else n_moe - 1}, token (b=0, s=0) -> "
        f"experts {ep[0, 0, -1].tolist()}"
    )
    return d


def coactivation(payloads):
    """Per-layer expert co-activation counts summed over every shard."""
    print("\n=== co-activation ===")
    counts = defaultdict(lambda: None)
    tokens = defaultdict(int)
    for d in payloads:
        ep = d.get("expert_paths")
        if ep is None:
            continue
        layers = d.get("moe_layer_indices") or list(range(ep.shape[2]))
        n_exp = int(ep.max()) + 1
        for li, layer in enumerate(layers):
            tokens[layer] += ep.shape[0] * ep.shape[1]
            sel = ep[:, :, li, :].reshape(-1, ep.shape[3]).long()  # [tokens, topk]
            onehot = torch.zeros(sel.shape[0], n_exp)
            onehot.scatter_(1, sel, 1.0)
            mat = onehot.T @ onehot  # [n_exp, n_exp]
            prev = counts[layer]
            if prev is None or prev.shape[0] < n_exp:
                grown = torch.zeros(n_exp, n_exp)
                if prev is not None:
                    grown[: prev.shape[0], : prev.shape[0]] = prev
                prev = grown
            prev += mat
            counts[layer] = prev

    if not counts:
        print("  nothing to do (no expert_paths found)")
        return

    for layer in sorted(counts):
        mat = counts[layer]
        pairs = mat.clone()
        pairs.fill_diagonal_(0)
        load = mat.diagonal()
        n_exp = mat.shape[0]
        # Strongest pair and how skewed the per-expert load is.
        flat = int(pairs.argmax())
        i, j = flat // n_exp, flat % n_exp
        off_diag_mass = pairs.sum() / 2
        print(
            f"  layer {layer:3d}: tokens={tokens[layer]} experts={n_exp} "
            f"load min/max={int(load.min())}/{int(load.max())} "
            f"top pair=({i},{j}) x{int(pairs[i, j])} "
            f"pair-mass={int(off_diag_mass)}"
        )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("paths", nargs="+", help="Capture directory, shard file(s), or glob(s).")
    ap.add_argument(
        "--max-files", type=int, default=4, help="How many shards to describe in full (default 4)."
    )
    ap.add_argument(
        "--coactivation",
        action="store_true",
        help="Also accumulate per-layer expert co-activation counts over ALL matched shards.",
    )
    args = ap.parse_args()

    files = collect(args.paths)
    if not files:
        print("no capture shards matched", file=sys.stderr)
        sys.exit(1)

    iters, mbs, ranks = set(), set(), set()
    for f in files:
        m = SHARD_RE.search(os.path.basename(f))
        if m:
            iters.add(int(m.group(1)))
            mbs.add(int(m.group(2)))
            ranks.add(int(m.group(3)))
    print(
        f"{len(files)} shard(s): iters={sorted(iters)} "
        f"micro_batches={sorted(mbs)} ranks={sorted(ranks)}"
    )
    if iters and len(files) != len(iters) * len(mbs) * len(ranks):
        print("[warn] shard set is ragged - some (iter, mb, rank) combinations are missing")

    described = [describe(f) for f in files[: args.max_files]]
    if len(files) > args.max_files:
        print(
            f"\n... {len(files) - args.max_files} more shard(s) not described "
            f"(raise --max-files to see them)"
        )

    if args.coactivation:
        payloads = described + [
            torch.load(f, weights_only=False, map_location="cpu") for f in files[args.max_files :]
        ]
        coactivation(payloads)


if __name__ == "__main__":
    main()
