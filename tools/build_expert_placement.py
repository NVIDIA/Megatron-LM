# Copyright (c) 2026, Zyphra Technologies Inc. All rights reserved.
"""Co-activation-based expert placement -> expert_placement_per_layer.json (offline tool).

Uses *within-layer* co-activation, read from route captures written by
``megatron/core/capture.py`` (MOE_ROUTE_CAPTURE=1), to decide for each MoE layer which
experts should be **co-located on the same EP rank**; the runtime consumes the
result through ``--moe-expert-placement`` (parsed in megatron/training/arguments.py,
applied in megatron/core/transformer/moe/moe_layer.py).

Why co-locate co-activated experts
----------------------------------
``coactivation[L][e1, e2] = P(e2 in top-k | e1 in top-k)`` (the notebook's
within-layer measure).  When a token's top-k experts live on the SAME device the
dispatch all-to-all sends that token to one rank instead of several, so grouping
frequently co-selected experts onto one rank shrinks EP communication.  We treat
the symmetric co-selection count matrix as an affinity graph and partition the
experts into ``ep_size`` equal-size groups that maximise intra-group weight; each
group becomes one rank's contiguous slot block.

Output (consumed by ``--moe-expert-placement-path``)
----------------------------------------------------
``_parse_moe_expert_placement`` expects an object with a rank-major permutation
of ``0..num_experts-1`` under ``default`` and/or per-layer under ``layers``,
keyed by the model ``layer_number``::

    {
      "num_experts": 128,
      "default": [ ...perm... ],                 # fallback for uncaptured layers
      "layers": { "3": [ ...perm... ], "7": [...] }   # key = moe_layer_indices[m]
    }

This is a pure-CPU offline tool (numpy + torch, no ``megatron`` import):
    python tools/build_expert_placement.py $CAPTURE_DIR \
        --iters 2000-2009 --num-experts 128 --ep-size 8 \
        --out expert_placement_per_layer.json
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


def _winner_first(
    expert_paths: torch.Tensor, routing_probs: Optional[torch.Tensor]
) -> torch.Tensor:
    """Order each captured top-k list by descending routing probability."""
    if routing_probs is None:
        return expert_paths
    order = torch.argsort(routing_probs.float(), dim=-1, descending=True, stable=True)
    return torch.gather(expert_paths, -1, order)


def load_expert_paths(capture_file: str) -> torch.Tensor:
    """Load winner-first expert paths from a capture shard."""
    obj = torch.load(capture_file, map_location="cpu", weights_only=False)
    expert_paths = obj["expert_paths"].to(torch.int64)
    return _winner_first(expert_paths, obj.get("routing_probs"))


def resolve_capture_files(capture_dir: str, iters: Sequence[int]) -> List[str]:
    """Return all capture shards for the requested training iterations."""
    prefixes = tuple(f"iter_{iteration:08d}" for iteration in iters)
    names = sorted(name for name in os.listdir(capture_dir) if name.startswith(prefixes))
    if not names:
        raise FileNotFoundError(f"No capture shards for iters {list(iters)} in {capture_dir}")
    return [os.path.join(capture_dir, name) for name in names]


# ---------------------------------------------------------------------------
# 1. Co-activation counts (mirrors the notebook's `coactivation`)
# ---------------------------------------------------------------------------


def coactivation_counts(paths: torch.Tensor, layer: int, n_experts: int) -> np.ndarray:
    """counts[e1, e2] = # tokens whose top-k at ``layer`` contains BOTH e1 and e2.
    Symmetric; the diagonal is the number of tokens that selected each expert."""
    topk = paths.shape[3]
    sel = paths[:, :, layer, :].reshape(-1, topk).numpy().astype(np.int64)  # [tokens, k]
    pid = (sel[:, :, None] * n_experts + sel[:, None, :]).reshape(-1)
    return np.bincount(pid, minlength=n_experts * n_experts).reshape(n_experts, n_experts)


def build_coactivation(
    capture_dir: str, iters: Sequence[int], num_experts: int
) -> Tuple[np.ndarray, List[int]]:
    """Aggregate per-layer co-activation counts across all shards for ``iters``.

    Returns (counts[M, E, E] int64, moe_layer_indices).  All shards must share
    the same ``moe_layer_indices`` (true for PP=1 or single-stage captures);
    otherwise position ``m`` would mean a different model layer in different
    shards, so we assert rather than silently mix them.
    """
    files = resolve_capture_files(capture_dir, iters)
    print(f"total files : {len(files)}")

    ref_layers: Optional[List[int]] = None
    counts: Optional[np.ndarray] = None
    for f in files:
        obj = torch.load(f, map_location="cpu", weights_only=False)
        layers = list(obj.get("moe_layer_indices", []))
        paths = load_expert_paths(f)
        cap_experts = int(paths.max().item()) + 1
        if cap_experts < num_experts:
            raise ValueError(
                f"{f}: capture has {cap_experts} experts, < --num-experts {num_experts}."
            )
        if ref_layers is None:
            ref_layers = layers or list(range(paths.shape[2]))
            counts = np.zeros((paths.shape[2], num_experts, num_experts), np.int64)
        elif layers and layers != ref_layers:
            raise ValueError(
                "Capture shards disagree on moe_layer_indices "
                f"({layers} vs {ref_layers}); per-layer placement needs a single "
                "PP stage. Filter shards to one pp_rank or capture with PP=1."
            )
        for m in range(counts.shape[0]):
            # Slice to the routed experts (drops the MOD skip expert if present).
            counts[m] += coactivation_counts(paths, m, cap_experts)[:num_experts, :num_experts]
        del paths
    assert counts is not None and ref_layers is not None
    return counts, ref_layers


# ---------------------------------------------------------------------------
# 2. Balanced affinity partition -> rank-major permutation
# ---------------------------------------------------------------------------


def kl_partition(
    coact: np.ndarray, ep_size: int, max_iter: Optional[int] = None, max_load_ratio: float = 0.0
) -> np.ndarray:
    """Kernighan-Lin balanced partition of the co-activation graph.

    Faithful port of the notebook's ``kl_partition`` (the ``coplace`` builder):
    start from the contiguous round-robin placement, then repeatedly apply the
    single best expert-swap between two ranks that most increases intra-rank
    co-activation weight, for ``max_iter`` (default ``3*E``) steps.  Swaps
    preserve group sizes, so the result stays perfectly balanced in expert COUNT.

    ``max_load_ratio > 0`` additionally balances token LOAD. The diagonal of
    ``coact`` is the number of tokens that selected each expert, so a rank's
    load is the sum of its members' diagonals. A swap is accepted only if both
    ranks end up at or below ``max_load_ratio * mean_rank_load``, or if it does
    not raise the larger of the two ranks' loads (so an initially over-full
    rank can still be repaired). Without this, grouping co-activated (= popular)
    experts piles load onto a few ranks; the all-to-all and the expert GEMMs are
    then bounded by the hottest rank, which cancels the wire-row savings.

    Returns ``rank[e]`` = the EP rank expert ``e`` is assigned to (NOT a
    permutation -- see :func:`rank_assignment_to_order`).
    """
    E = coact.shape[0]
    assert E % ep_size == 0, f"num_experts {E} not divisible by ep_size {ep_size}"
    cap = E // ep_size
    if max_iter is None:
        max_iter = 3 * E

    W = coact.astype(np.float64).copy()
    load = np.diag(W).copy()  # tokens that selected each expert
    np.fill_diagonal(W, 0.0)  # self-count must not drive grouping
    rank = (np.arange(E) // cap).copy()
    mem = {r: list(np.where(rank == r)[0]) for r in range(ep_size)}
    # WR[e, r] = total co-activation weight from expert e to experts on rank r.
    WR = np.stack([W[:, mem[r]].sum(1) for r in range(ep_size)], axis=1)
    rank_load = np.array([load[mem[r]].sum() for r in range(ep_size)])
    load_cap = max_load_ratio * rank_load.mean() if max_load_ratio > 0 else None

    for _ in range(max_iter):
        best_gain, best = 1e-9, None
        for ra in range(ep_size):
            A = mem[ra]
            for rb in range(ra + 1, ep_size):
                B = mem[rb]
                # gain of swapping a in A with b in B (|A| x |B| matrix)
                G = (
                    (WR[np.ix_(A, [rb])] - WR[np.ix_(A, [ra])])
                    + (WR[np.ix_(B, [ra])] - WR[np.ix_(B, [rb])]).T
                    - 2 * W[np.ix_(A, B)]
                )
                if load_cap is not None:
                    la = load[A][:, None]
                    lb = load[B][None, :]
                    new_a = rank_load[ra] - la + lb
                    new_b = rank_load[rb] + la - lb
                    within_cap = (new_a <= load_cap) & (new_b <= load_cap)
                    not_worse = np.maximum(new_a, new_b) <= max(rank_load[ra], rank_load[rb])
                    G = np.where(within_cap | not_worse, G, -np.inf)
                i, j = np.unravel_index(np.argmax(G), G.shape)
                if G[i, j] > best_gain:
                    best_gain, best = G[i, j], (A[i], B[j], ra, rb)
        if best is None:
            break
        a, b, ra, rb = best
        rank[a], rank[b] = rb, ra
        mem[ra].remove(a)
        mem[ra].append(b)
        mem[rb].remove(b)
        mem[rb].append(a)
        WR[:, ra] += W[:, b] - W[:, a]
        WR[:, rb] += W[:, a] - W[:, b]
        rank_load[ra] += load[b] - load[a]
        rank_load[rb] += load[a] - load[b]

    return rank


def load_imbalance(coact: np.ndarray, order: List[int], ep_size: int) -> float:
    """max / mean per-rank token load of a rank-major ``order`` (1.0 = perfectly even).

    Per-expert load is the diagonal of ``coact`` (tokens that selected the expert)."""
    E = coact.shape[0]
    cap = E // ep_size
    load = np.diag(coact).astype(np.float64)
    per_rank = np.array([load[order[r * cap : (r + 1) * cap]].sum() for r in range(ep_size)])
    return float(per_rank.max() / per_rank.mean()) if per_rank.mean() > 0 else 1.0


def rank_assignment_to_order(rank: np.ndarray, ep_size: int) -> List[int]:
    """Rank-major slot permutation from a ``rank[e]`` assignment: experts on rank
    0 first (ascending id), then rank 1, ...  Matches how the notebook reads
    ``np.where(coplace[L] == r)`` per rank."""
    order: List[int] = []
    for r in range(ep_size):
        order.extend(int(e) for e in np.where(rank == r)[0])
    assert sorted(order) == list(range(rank.size)), "placement is not a permutation"
    return order


def coactivation_placement(
    coact: np.ndarray, ep_size: int, nodes: int = 1, max_load_ratio: float = 0.0
) -> List[int]:
    """Co-activation placement as a rank-major permutation (KL partition).

    ``nodes > 1`` makes the partition hierarchical (node-aware): experts are first
    split into ``nodes`` balanced groups that maximise intra-NODE co-activation, then
    each node group is split into its ``ep_size // nodes`` ranks.  EP ranks are
    assumed node-major (rank r lives on node r // (ep_size // nodes)), which is how
    torchrun lays out a multi-node job.  Cross-node wire rows are the volume-priced
    ones on the NIC fabric, so the node level is optimised first.

    ``max_load_ratio`` is forwarded to :func:`kl_partition` (token-load cap per rank).
    """
    if nodes <= 1:
        return rank_assignment_to_order(
            kl_partition(coact, ep_size, max_load_ratio=max_load_ratio), ep_size
        )
    assert ep_size % nodes == 0, f"ep_size {ep_size} not divisible by nodes {nodes}"
    per_node = ep_size // nodes
    node_of = kl_partition(coact, nodes, max_load_ratio=max_load_ratio)
    rank = np.empty(coact.shape[0], dtype=np.int64)
    for n in range(nodes):
        members = np.where(node_of == n)[0]
        sub = coact[np.ix_(members, members)]
        local_rank = kl_partition(sub, per_node, max_load_ratio=max_load_ratio)
        rank[members] = n * per_node + local_rank
    return rank_assignment_to_order(rank, ep_size)


def node_quality(coact: np.ndarray, order: List[int], ep_size: int, nodes: int) -> float:
    """Fraction of co-activation weight kept within NODE groups (node-major ranks)."""
    if nodes <= 1:
        return placement_quality(coact, order, ep_size)
    return placement_quality(coact, order, nodes)


def placement_quality(coact: np.ndarray, order: List[int], ep_size: int) -> float:
    """Fraction of co-activation weight kept *within* rank groups (higher is
    better; 1.0 = every co-selected pair co-located, identity-order baseline is
    whatever contiguous blocks happen to capture)."""
    E = coact.shape[0]
    cap = E // ep_size
    W = coact.astype(np.float64).copy()
    np.fill_diagonal(W, 0.0)
    total = W.sum()
    if total == 0:
        return 0.0
    kept = 0.0
    for r in range(ep_size):
        g = order[r * cap : (r + 1) * cap]
        kept += W[np.ix_(g, g)].sum()
    return float(kept / total)


# ---------------------------------------------------------------------------
# 3. Build the JSON
# ---------------------------------------------------------------------------


def build_placement_json(
    capture_dir: str,
    iters: Sequence[int],
    num_experts: int,
    ep_size: int,
    emit_default: bool = True,
    nodes: int = 1,
    max_load_ratio: float = 0.0,
) -> Dict:
    counts, moe_layer_indices = build_coactivation(capture_dir, iters, num_experts)
    n_moe = counts.shape[0]

    out: Dict[str, object] = {"num_experts": num_experts, "layers": {}}
    if nodes > 1:
        out["nodes"] = nodes
    if max_load_ratio > 0:
        out["max_load_ratio"] = max_load_ratio
    contiguous = list(range(num_experts))
    print(
        f"\nper-layer placement (quality = intra-rank co-activation kept"
        + (f"; node = intra-node kept, {nodes} nodes" if nodes > 1 else "")
        + "; load = max/mean per-rank token load, contiguous placement in parentheses):"
    )
    for m in range(n_moe):
        order = coactivation_placement(counts[m], ep_size, nodes, max_load_ratio)
        q = placement_quality(counts[m], order, ep_size)
        layer_number = int(moe_layer_indices[m])
        out["layers"][str(layer_number)] = order
        extra = (
            f"  node {node_quality(counts[m], order, ep_size, nodes)*100:5.1f}%"
            if nodes > 1
            else ""
        )
        print(
            f"  layer {layer_number:>3} (moe pos {m:>2}): quality {q*100:5.1f}%{extra}"
            f"  load {load_imbalance(counts[m], order, ep_size):.3f}"
            f" ({load_imbalance(counts[m], contiguous, ep_size):.3f})"
        )

    if emit_default:
        agg = counts.sum(axis=0)
        default = coactivation_placement(agg, ep_size, nodes, max_load_ratio)
        out["default"] = default
        print(
            f"  default (all layers summed): quality "
            f"{placement_quality(agg, default, ep_size)*100:5.1f}%"
            f"  load {load_imbalance(agg, default, ep_size):.3f}"
            f" ({load_imbalance(agg, contiguous, ep_size):.3f})"
        )
    return out


def _parse_iters(spec: str) -> List[int]:
    if "-" in spec:
        lo, hi = (int(x) for x in spec.split("-"))
        return list(range(lo, hi + 1))
    return [int(x) for x in spec.split(",")]


def _cli(argv: Optional[Sequence[str]] = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Build expert_placement_per_layer.json from CMoE capture co-activation."
    )
    ap.add_argument("capture_dir", help="dir with iter_*.pt capture shards")
    ap.add_argument("--iters", required=True, help="e.g. '2000-2009' or '2000,2001'")
    ap.add_argument("--num-experts", type=int, required=True, help="routed experts (--num-experts)")
    ap.add_argument("--ep-size", type=int, required=True, help="expert-model-parallel-size")
    ap.add_argument("--out", required=True, help="output .json path")
    ap.add_argument("--no-default", action="store_true", help="omit the 'default' fallback perm")
    ap.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="node-aware two-level partition: optimise intra-node co-activation first "
        "(EP ranks node-major, ep_size/nodes ranks per node)",
    )
    ap.add_argument(
        "--max-load-ratio",
        type=float,
        default=1.1,
        help="cap each rank's token load (sum of its experts' selection counts) at this "
        "multiple of the mean while partitioning; 0 disables load balancing (pure "
        "co-activation partition, which tends to pile popular experts onto a few ranks)",
    )
    args = ap.parse_args(argv)

    obj = build_placement_json(
        args.capture_dir,
        _parse_iters(args.iters),
        args.num_experts,
        args.ep_size,
        emit_default=not args.no_default,
        nodes=args.nodes,
        max_load_ratio=args.max_load_ratio,
    )
    with open(args.out, "w") as f:
        json.dump(obj, f, indent=2)
    print(
        f"\nsaved -> {args.out}  "
        f"({len(obj['layers'])} layers, num_experts={args.num_experts}, ep_size={args.ep_size})"
    )


if __name__ == "__main__":
    _cli()
