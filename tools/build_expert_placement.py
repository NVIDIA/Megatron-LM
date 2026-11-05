# Copyright (c) 2026, Zyphra Technologies Inc. All rights reserved.
"""Co-activation-based expert placement -> expert_placement_per_layer.json (offline tool).

Uses *within-layer* co-activation, read from the router traces written by
``--moe-routing-trace-path`` (from a PP=1 run), to decide for each MoE layer which
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
      "layers": { "3": [ ...perm... ], "7": [...] }   # key = layer_number
    }

This is a pure-CPU offline tool (numpy, no ``megatron`` import):
    python tools/build_expert_placement.py $TRACE_DIR \
        --iters 2000-2009 --num-experts 128 --ep-size 8 \
        --out expert_placement_per_layer.json
"""

from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def build_coactivation(
    trace_dir: str, iters: Optional[Sequence[int]], num_experts: int
) -> Tuple[np.ndarray, List[int]]:
    """Per-layer co-activation counts from ``--moe-routing-trace-path`` traces.

    counts[m, e1, e2] = # tokens whose top-k at layer ``layers[m]`` contains both e1
    and e2; the diagonal is the number of tokens that selected each expert.
    """
    counts: Dict[int, np.ndarray] = {}
    for path in sorted(glob.glob(os.path.join(trace_dir, "router_trace_rank*.jsonl"))):
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("block", "decoder") != "decoder" or (iters and rec["step"] not in iters):
                    continue
                sel = np.asarray(rec["top_indices"], dtype=np.int64).reshape(-1, rec["topk"])
                if (sel >= num_experts).any():
                    raise ValueError(f"{path}: expert ids exceed --num-experts {num_experts}.")
                pid = (sel[:, :, None] * num_experts + sel[:, None, :]).reshape(-1)
                # Trace layers are 0-based decoder.layers indices; placements use layer_number.
                layer = rec["layer"] + 1
                counts[layer] = counts.get(layer, 0) + np.bincount(pid, minlength=num_experts**2)
    if not counts:
        raise FileNotFoundError(f"No decoder router traces for steps {iters} in {trace_dir}")
    layers = sorted(counts)
    return np.stack([counts[l].reshape(num_experts, num_experts) for l in layers]), layers


# ---------------------------------------------------------------------------
# 2. Balanced affinity partition -> rank-major permutation
# ---------------------------------------------------------------------------


def kl_partition(
    coact: np.ndarray,
    ep_size: int,
    max_iter: Optional[int] = None,
    max_load_ratio: float = 0.0,
    mean_load: Optional[float] = None,
) -> np.ndarray:
    """Kernighan-Lin balanced partition of the co-activation graph.

    Faithful port of the notebook's ``kl_partition`` (the ``coplace`` builder):
    start from the contiguous round-robin placement, then repeatedly apply the
    single best expert-swap between two ranks that most increases intra-rank
    co-activation weight, for ``max_iter`` (default ``3*E``) steps.  Swaps
    preserve group sizes, so the result stays perfectly balanced in expert COUNT.

    ``max_load_ratio > 0`` additionally balances token LOAD, starting from a
    load-balanced grouping. The diagonal of ``coact`` is the number of tokens
    that selected each expert, so a rank's
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
    if max_load_ratio > 0:
        snake = np.concatenate([np.arange(ep_size), np.arange(ep_size)[::-1]])
        rank[np.argsort(-load, kind="stable")] = np.resize(snake, E)
    mem = {r: list(np.where(rank == r)[0]) for r in range(ep_size)}
    # WR[e, r] = total co-activation weight from expert e to experts on rank r.
    WR = np.stack([W[:, mem[r]].sum(1) for r in range(ep_size)], axis=1)
    rank_load = np.array([load[mem[r]].sum() for r in range(ep_size)])
    if mean_load is None:
        mean_load = rank_load.mean()
    load_cap = max_load_ratio * mean_load if max_load_ratio > 0 else None

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
        local_rank = kl_partition(
            sub, per_node, max_load_ratio=max_load_ratio, mean_load=np.trace(coact) / ep_size
        )
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
    trace_dir: str,
    iters: Optional[Sequence[int]],
    num_experts: int,
    ep_size: int,
    emit_default: bool = True,
    nodes: int = 1,
    max_load_ratio: float = 0.0,
) -> Dict:
    counts, moe_layer_indices = build_coactivation(trace_dir, iters, num_experts)
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
        description="Build expert_placement_per_layer.json from router-trace co-activation."
    )
    ap.add_argument("trace_dir", help="--moe-routing-trace-path dir of a PP=1 run")
    ap.add_argument("--iters", default=None, help="trace steps, e.g. '2000-2009' (default: all)")
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
        args.trace_dir,
        _parse_iters(args.iters) if args.iters else None,
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
