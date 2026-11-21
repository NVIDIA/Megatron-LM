<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Pitfalls and rejected approaches

Things that look reasonable but produce wrong numbers. Documented here so future-you
doesn't re-discover them.

## 1. Dividing wall-window by N_iter

```
per_iter ≈ (last_kernel.end − first_kernel.start) / num_iters     ← WRONG
```

The wall window includes warmup, profiler startup, post-iter teardown, and any
idle time at the head/tail. This can underestimate per-iter time and dilute Δ.
Real case: this method reported a 5.7% gap when the actual gap was 19%.

**Use**: an iter-anchor (NCCL/sync/optimizer kernel timestamps) and measure
`anchor[i+1].start − anchor[i].start`. See `scripts/iter_anchor.py`.

## 2. Summing kernel durations to estimate wall time

```
total_busy ≈ SUM(end - start) over all kernels    ← WRONG
```

Modern training/inference uses multiple streams (compute + comm, fwd/bwd
overlap, multi-stream fan-out). Summing overstates wall time 2–4× because it
double-counts intervals when ≥2 streams are busy simultaneously.

**Use**: interval union (`_lib.union_total_ns`). Sum is fine for *per-category
work volume* (Step 3) but never for wall-time attribution (Steps 2, 5, 6).

## 3. Same iter count ≠ same op count per iter

Two implementations of the same model can issue different numbers of kernels
per iter (different fusion, different parallelism, different bucketing). The
user telling you "both ran 10 iters" doesn't mean each iter has the same
internal structure. Real case: 18 vs 23 AllGathers per iter from different
distributed-optimizer bucket sizes.

**Use**: verify anchor count constancy per iter, on each profile
independently. If it differs between profiles, that's a finding to investigate
(Step 6 RC) rather than a methodology bug.

## 4. Single-main-stream anchoring under stream fan-out

If implementation A places all compute on one big stream and implementation B
fans out across many small streams, anchoring on "the main compute stream"
will yield mismatched anchor counts (e.g. 1138 vs 540) — module-slicing then
can't pair up the windows.

**Use**: global anchor sequence across **all compute streams** (NCCL excluded
by name, not by stream — comm streams sometimes carry non-comm work too).
Anchor counts then match because they reflect the model's matmul count, not
the dispatcher's stream choice. See `scripts/module_slice.py`.

## 5. Op-group attribution under heavy fusion

Op-group is the natural Step 4 default: "rmsnorm took X ms on flat, Y ms on
non-flat". But if implementation A has a single custom kernel
`_qkv_split_norm_rope_kernel` doing 3 ops while B runs separate rmsnorm + rope
+ clone, then:

- A's `rmsnorm` row shows 0 ms (it's hidden in the fused kernel).
- B's `rmsnorm` row shows the full cost.
- Reporting "A saves X ms in rmsnorm" misrepresents the gap.

Op-group breakdown also has a hard problem of attributing small ambiguous
kernels (a `vectorized_add` could belong to multiple op-groups; a single
misallocation can swing a row by 10–40 ms).

**Use**: op-group as the default; switch to module-slicing when the share of
non-anchor compute time in *custom-fused kernels* (heuristic: names matching
`_*_fused_*` or 2+ op tokens in `triton_*_fused_*`) exceeds 10% in either
profile. `scripts/categorize.py` emits `fused_share_of_residual_pct` and
`module_slicing_recommended` to automate this.

## 6. Phenomenon ≠ root cause

"The cuBLAS heuristic picks a different kernel" is not a root cause — it's a
phenomenon restatement. Library heuristics are deterministic: same inputs →
same output. If two callers get different results, at least one input differs.
**Find which input.**

Similarly: "more kernel launches", "different bucketing", "different SDPA
backend" — all phenomena. The cause is the specific config/env-var/code-path
that produces the different choice.

**Use**: for each Δ, drill until the answer is "implementation A calls X with
Y; B calls X with Z; the heuristic at <file:line> picks differently because of
Y vs Z". Step 6 of the SKILL prompts for source-level evidence, mechanism, and
actionable change.

## 7. YAML as golden reference

It's tempting to ship a "correct" YAML taxonomy. There isn't one. The
taxonomy must match the workload — Triton kernel names differ between
torch.compile versions; custom kernels have idiosyncratic names. A regex
that's right for workload A may mis-classify workload B.

**Use**: treat the bundled `taxonomy_template.yml` as a starting point. For
every Step 3 / 4 run, inspect the matched-kernel list per category (false
positives) and the uncategorized list (false negatives), then iterate the
regexes. Save the final YAML alongside the report.

## 8. Naming `_h_badd_` as "bias-add"

The nvjet/cuBLAS kernel-family suffixes (`_h_bz_`, `_h_badd_`, `_NTT`, `_TNT`,
etc.) are internal cuBLAS heuristic names; their meaning is **not publicly
documented**. Treating them as if they encode op semantics (e.g. "badd =
bias-add epilogue") can be wrong — for example, a wgrad GEMM with `bias=None`
and `accumulate=True` may pick `_h_badd_*` for the FP32-RMW accumulator path,
not for a bias epilogue.

**Use**: the suffix flip is a *symptom*. The cause is whichever cuBLAS input
differs between the two callers (epilogue arg, output dtype, accumulate flag,
alignment, recipe). Read the actual `cublasLtMatmul` call args, not the kernel
name.

## 9. Treating "graph break" as proven from kernel pattern alone

If implementation A has one big fused Triton kernel and B has many small
Triton + TE kernels, the natural conclusion is "torch.compile graph-breaks at
TE C++ calls in B." That's plausible — but the kernel pattern alone doesn't
prove it. To verify, read `transformer_engine/pytorch/jit.py` for the
`no_torch_dynamo` / `torch._dynamo.disable` decorators applied to TE modules;
or capture `TORCH_LOGS=graph_breaks` from a runtime invocation.

**Use**: in Step 6, include source decorator evidence or runtime log evidence;
otherwise mark the mechanism as "inferred from kernel pattern" in the
verification-status table.

## 10. Forgetting the verification-status table at the end of Step 6

Without it, readers can't tell which claims are source-verified, which are
trace-inferred, and which are unverified guesses. Every RC's mechanism needs
a status row. Required, not optional.

## 11. Single-rank assumption under TP/PP/SP

With tensor/pipeline/sequence parallelism, the profiled rank only sees the
collectives and compute that rank participates in. An anchor that exists on
rank 0 may not exist on rank 1. Per-iter time on rank 0 may differ from rank
1 (bubble, imbalance). The user telling you "rank 0" doesn't mean rank 0's
view is representative.

**Use**: check the anchor type's frequency. If it's `~N` per iter with low
jitter, fine. If it's wildly variable or missing, the profile may be
non-representative or the anchor is wrong for this rank.
