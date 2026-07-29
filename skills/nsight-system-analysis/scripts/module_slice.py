#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Module-slice a profile: anchor on every GEMM+MHA kernel globally across all
compute streams, then sum (or union) the non-NCCL work in each inter-anchor window.

Designed for Step 4 when op-group attribution is unreliable due to heavy fusion.
Both implementations should produce roughly equal anchor counts (same model →
same number of matmuls); the resulting windows are apples-to-apples.

Usage:
  python module_slice.py <sqlite> --yaml taxonomy.yml --windows windows.json
  python module_slice.py <sqlite> --yaml taxonomy.yml --windows windows.json \
        --anchor-categories gemm,mha,flash_attn,cudnn_sdpa

Output (stdout JSON):
  {
    "anchor_count_total": 1530,
    "anchor_count_per_iter": [1530, 1530, ...],
    "anchor_count_per_iter_constant": true,
    "anchor_overlap_pct": 33.2,
    "grouped_windows": [
      {
        "signature": "gemm_TNT -> gemm_TNT",
        "count_per_iter": 38,
        "median_total_union_ms_per_iter": 30.4,
        "median_total_sum_ms_per_iter": 30.6
      },
      ...
    ],
    "iter_total_anchor_ms_median": 222.5,
    "iter_total_window_union_ms_median": 51.9,
    "iter_total_window_sum_ms_median": 53.5
  }

The `signature` is
`<left_anchor_category_or_shape> -> <right_anchor_category_or_shape>`,
where the "shape" is the part of the kernel name after the category-defining
substring. This keeps related anchors grouped while distinguishing different
matmul shapes. If shapes are too noisy, use `--signature-mode category` for the
coarser "gemm -> gemm" / "gemm -> mha" grouping.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from statistics import median

from _lib import (
    classify,
    clip,
    load_taxonomy,
    open_sqlite,
    union_total_ns,
    write_json,
)


def load_windows(path: str) -> list[tuple[int, int]]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "windows_ns" in data:
        data = data["windows_ns"]
    return [(int(w[0]), int(w[1])) for w in data]


def anchor_signature(name: str, cat: str, mode: str) -> str:
    """Build a signature string for an anchor kernel.

    mode='category': just the category (e.g. 'gemm', 'mha').
    mode='shape': category + the kernel-shape tail (e.g. 'gemm_128x256_TNT').
    """
    if mode == "category":
        return cat
    # Pull out a shape hint from the kernel name.
    # nvjet: nvjet_sm103_qqtst_128x256_128x6_2x2f_2cta_h_bz_..._NTT
    m = re.search(r"(\d+x\d+(?:_\d+x\d+)?)", name)
    shape = m.group(1) if m else ""
    # Pull out NTT/NNT/TNT/TNN suffix if present
    m2 = re.search(r"_([NT]{2,4})(?:\b|$)", name)
    suffix = m2.group(1) if m2 else ""
    parts = [cat]
    if shape:
        parts.append(shape)
    if suffix:
        parts.append(suffix)
    return "_".join(parts)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("sqlite")
    p.add_argument("--yaml", required=True)
    p.add_argument("--windows", required=True)
    p.add_argument(
        "--anchor-categories",
        default="gemm,flash_attn,cudnn_sdpa,mha,conv",
        help="Comma-separated YAML categories whose kernels serve as anchors",
    )
    p.add_argument(
        "--signature-mode",
        choices=["category", "shape"],
        default="shape",
        help="Granularity of anchor-pair signatures used for window grouping",
    )
    args = p.parse_args()

    taxonomy = load_taxonomy(args.yaml)
    windows = load_windows(args.windows)
    anchor_cats = {c.strip() for c in args.anchor_categories.split(",") if c.strip()}
    nccl_rx = taxonomy.get("nccl")

    anchor_count_per_iter: list[int] = []
    overlaps_per_iter: list[int] = []
    iter_anchor_ms: list[float] = []
    iter_window_union_ms: list[float] = []
    iter_window_sum_ms: list[float] = []

    # signature -> list[ (window_union_ns, window_sum_ns) ]; one entry per
    # (window instance) accumulated across iters.
    sig_window_union: dict[str, list[int]] = defaultdict(list)
    sig_window_sum: dict[str, list[int]] = defaultdict(list)
    sig_count_per_iter: dict[str, list[int]] = defaultdict(list)

    with open_sqlite(args.sqlite) as con:
        for lo, hi in windows:
            # Pull all kernels in the iter window.
            rows = con.execute(
                """
                SELECT k.start, k.end, k.streamId,
                COALESCE((SELECT value FROM StringIds WHERE id=k.demangledName), '')
                FROM CUPTI_ACTIVITY_KIND_KERNEL k
                WHERE k.end > ? AND k.start < ?
                ORDER BY k.start
                """,
                (lo, hi),
            ).fetchall()

            anchors = []  # list of (start, end, category, name)
            non_anchor_intervals_by_stream: dict[int, list[tuple[int, int]]] = (
                defaultdict(list)
            )

            for s, e, sid, name in rows:
                cs, ce = max(s, lo), min(e, hi)
                if ce <= cs:
                    continue
                cat = classify(name, taxonomy) or "_uncat_"
                is_nccl = cat == "nccl" or (nccl_rx and nccl_rx.search(name))
                if is_nccl:
                    continue  # NCCL excluded entirely from module-slicing
                if cat in anchor_cats:
                    anchors.append((cs, ce, cat, name))
                else:
                    non_anchor_intervals_by_stream[sid].append((cs, ce))

            anchors.sort(key=lambda a: a[0])
            anchor_count_per_iter.append(len(anchors))

            # Anchor time + window times.
            anchor_total_ns = sum(e - s for s, e, _, _ in anchors)
            iter_anchor_ms.append(anchor_total_ns / 1e6)

            # Group non-anchor intervals across all streams into one flat list
            # (we want the union across streams = wall-time GPU busy in the window).
            all_non_anchor = []
            for sid, ivs in non_anchor_intervals_by_stream.items():
                all_non_anchor.extend(ivs)

            overlaps = 0
            window_union_ns_iter = 0
            window_sum_ns_iter = 0
            sig_count_this_iter: dict[str, int] = defaultdict(int)

            # Iterate inter-anchor windows. To avoid losing time inside
            # overlap-collapsed window groups, we maintain a "cursor" which is
            # the running max(end) of the most recent anchor group. The next
            # window begins at the cursor, not at the immediately preceding
            # anchor's end. This way: if anchors k..k+M all overlap, we
            # collapse them into a single anchor group ending at max(end_k..k+M),
            # and the window from there to anchor[k+M+1].start carries any
            # non-anchor work without being dropped.
            n = len(anchors)
            cursor = anchors[0][1] if n else 0
            k = 0
            while k < n - 1:
                a_left = anchors[k]
                # Advance cursor past any overlapping run of anchors that
                # collectively span the next anchor's start.
                end_of_group = a_left[1]
                while k + 1 < n and anchors[k + 1][0] < end_of_group:
                    end_of_group = max(end_of_group, anchors[k + 1][1])
                    overlaps += 1
                    k += 1
                if k + 1 >= n:
                    break
                a_right = anchors[k + 1]
                w_lo = max(end_of_group, cursor)
                w_hi = a_right[0]
                if w_hi > w_lo:
                    sig = (
                        anchor_signature(a_left[3], a_left[2], args.signature_mode)
                        + " -> "
                        + anchor_signature(a_right[3], a_right[2], args.signature_mode)
                    )
                    clipped = clip(all_non_anchor, w_lo, w_hi)
                    w_sum_ns = sum(e - s for s, e in clipped)
                    w_union_ns = union_total_ns(clipped)
                    sig_window_sum[sig].append(w_sum_ns)
                    sig_window_union[sig].append(w_union_ns)
                    sig_count_this_iter[sig] += 1
                    window_sum_ns_iter += w_sum_ns
                    window_union_ns_iter += w_union_ns
                cursor = max(end_of_group, a_right[1])
                k += 1

            # Tail: include any non-anchor work between the last anchor and
            # the iter window end. This is the "post-last-anchor" region.
            if n:
                last_end = anchors[-1][1]
                w_lo = max(last_end, cursor)
                w_hi = hi
                if w_hi > w_lo:
                    sig = (
                        anchor_signature(
                            anchors[-1][3], anchors[-1][2], args.signature_mode
                        )
                        + " -> [iter_end]"
                    )
                    clipped = clip(all_non_anchor, w_lo, w_hi)
                    w_sum_ns = sum(e - s for s, e in clipped)
                    w_union_ns = union_total_ns(clipped)
                    sig_window_sum[sig].append(w_sum_ns)
                    sig_window_union[sig].append(w_union_ns)
                    sig_count_this_iter[sig] += 1
                    window_sum_ns_iter += w_sum_ns
                    window_union_ns_iter += w_union_ns
            # Pre-first-anchor region: also include.
            if n:
                w_lo = lo
                w_hi = anchors[0][0]
                if w_hi > w_lo:
                    sig = "[iter_start] -> " + anchor_signature(
                        anchors[0][3], anchors[0][2], args.signature_mode
                    )
                    clipped = clip(all_non_anchor, w_lo, w_hi)
                    w_sum_ns = sum(e - s for s, e in clipped)
                    w_union_ns = union_total_ns(clipped)
                    sig_window_sum[sig].append(w_sum_ns)
                    sig_window_union[sig].append(w_union_ns)
                    sig_count_this_iter[sig] += 1
                    window_sum_ns_iter += w_sum_ns
                    window_union_ns_iter += w_union_ns

            for sig in sig_count_this_iter:
                sig_count_per_iter[sig].append(sig_count_this_iter[sig])
            overlaps_per_iter.append(overlaps)
            iter_window_union_ms.append(window_union_ns_iter / 1e6)
            iter_window_sum_ms.append(window_sum_ns_iter / 1e6)

    n_iters = len(windows)
    n_anchors = anchor_count_per_iter[0] if anchor_count_per_iter else 0
    constant = len(set(anchor_count_per_iter)) == 1
    overlap_pct = (
        100
        * sum(overlaps_per_iter)
        / max(1, sum(a - 1 for a in anchor_count_per_iter if a > 0))
        if any(anchor_count_per_iter)
        else 0.0
    )

    # Aggregate per-signature: sum across windows in iter, then median across iters.
    grouped = []
    for sig in sig_window_union:
        total_union_per_iter = []
        total_sum_per_iter = []
        # We accumulated per-window samples; convert back to per-iter by chunking
        # using sig_count_per_iter.
        counts = sig_count_per_iter[sig]
        idx = 0
        for c in counts:
            u_slice = sig_window_union[sig][idx : idx + c]
            s_slice = sig_window_sum[sig][idx : idx + c]
            idx += c
            total_union_per_iter.append(sum(u_slice))
            total_sum_per_iter.append(sum(s_slice))
        # Pad with zeros if some iters have no instance of this sig.
        while len(total_union_per_iter) < n_iters:
            total_union_per_iter.append(0)
            total_sum_per_iter.append(0)
        grouped.append(
            {
                "signature": sig,
                "count_per_iter": round(median(counts) if counts else 0, 2),
                "median_union_ms_per_iter": round(
                    median(total_union_per_iter) / 1e6, 3
                ),
                "median_sum_ms_per_iter": round(median(total_sum_per_iter) / 1e6, 3),
            }
        )
    grouped.sort(key=lambda g: -g["median_union_ms_per_iter"])

    out = {
        "anchor_count_per_iter": anchor_count_per_iter,
        "anchor_count_constant_across_iters": constant,
        "anchor_count_per_iter_first": n_anchors,
        "anchor_overlap_pct": round(overlap_pct, 2),
        "iter_total_anchor_ms_median": round(
            median(iter_anchor_ms) if iter_anchor_ms else 0, 3
        ),
        "iter_total_window_union_ms_median": round(
            median(iter_window_union_ms) if iter_window_union_ms else 0, 3
        ),
        "iter_total_window_sum_ms_median": round(
            median(iter_window_sum_ms) if iter_window_sum_ms else 0, 3
        ),
        "grouped_windows": grouped,
        "notes": [
            "Use median_union_ms_per_iter as the primary number for "
            "wall-time attribution.",
            "median_sum_ms_per_iter is shown for comparison; sum overcounts "
            "when non-anchor kernels run in parallel on multiple compute streams.",
            "NCCL kernels are excluded entirely (by name match against the "
            "YAML's `nccl` category).",
        ],
    }
    write_json(out)


if __name__ == "__main__":
    main()
