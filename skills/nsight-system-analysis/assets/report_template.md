<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# <Workload name> — Performance <Gap | Analysis>

## Inputs

<!-- For comparative mode, list both profiles. For single-profile mode, list one. -->
- **Profile A** (fast / baseline): `<path>`
- **Profile B** (slow / variant): `<path>`
- Same config: <bs, precision, GPUs, iter count, rank>
- YAML taxonomy: `<path to verified yaml>`

## Step 1: Per-Iteration Time

### Anchor

<!-- Fill in which anchor was chosen and why. State the second-anchor cross-check. -->

| Anchor | A count | B count | Per-iter N (A / B) | Notes |
|---|---|---|---|---|
| <chosen anchor> | … | … | … | … |
| Cross-check anchor | … | … | … | agree to <X> ms |

### Per-iteration time

| Profile | Median (ms) | Min (ms) | Max (ms) |
|---|---|---|---|
| A | … | … | … |
| B | … | … | … |
| **Δ** | **… ms** | — | — |
| **Δ %** | **… %** | — | — |

**Headline**: <one-sentence summary, e.g. "B is X ms slower per iter (Y%)">

## Step 2: GPU Busy vs GPU Idle

GPU busy = interval-union of kernels + memcpys across all streams within each iter
window. GPU idle = `iter_time − busy` and is the CPU-bound portion.

| Metric | A | B |
|---|---|---|
| iter time (ms) | … | … |
| GPU busy (union) | … | … |
| GPU idle | … | … |
| GPU idle % | … | … |
| Longest single-stream union (any kernel) | … | … |
| Longest single-stream union (non-NCCL) | … | … |

### Δ (B − A)

| Metric | Δ | Share of iter Δ |
|---|---|---|
| iter time | … | 100% |
| GPU busy | … | … |
| GPU idle | … | … |

**Interpretation**: <is the gap GPU-busy-dominated (kernel-side) or
GPU-idle-dominated (CPU/dispatch-side)? Or both?>

## Step 3: Heavy-Compute Categories (GEMM / Conv / MHA)

Scope: only model-level compute that is essentially never fused into anything
else. Norm is in Step 4 (commonly fused).

| Category | A (ms/iter) | B (ms/iter) | Δ ms | Δ % |
|---|---|---|---|---|
| gemm | … | … | … | … |
| conv | … | … | … | … |
| mha (flash + cudnn_sdpa) | … | … | … | … |
| **Sum** | … | … | … | … |

**Interpretation**: <are GEMM/Conv/MHA roughly equal? If not, drill into per-shape
diff in Step 6.>

### YAML verification

YAML at `<path>` was iterated. Final state:
- Matched-kernel-list per category inspected; all matches confirmed.
- Uncategorized list contains <list of leftovers, none >1% of iter>.

## Step 4: <Op-Group Breakdown | Module-Slicing>

<!-- The script reports `fused_share_of_residual_pct` and `module_slicing_recommended`.
Use whichever mode applies, and STATE the decision metric and why you chose it.
Then DELETE the unused sub-section below (keep only the one matching your chosen
mode). -->

**Decision metric**: custom-fused kernel share of non-anchor compute time = <X%>
(threshold: 10%). Mode chosen: **<op-group | module-slicing>** because <reason>.

### Op-group mode (when fusion is light) — DELETE this section if using module-slicing

| Operator | A (ms/iter) | B (ms/iter) | Δ ms | Notes |
|---|---|---|---|---|
| <rmsnorm / layer_norm / rope / fp8_cast / elementwise / triton / ...> | … | … | … | … |
| **Sum (residual)** | … | … | … | |

### Module-slicing mode (when fusion is heavy) — DELETE this section if using op-group

Global anchor sequence: A had **<N_A>** anchors/iter, B had **<N_B>** (within
<X>%). Anchor overlaps (windows skipped): <Y%>.

| Window (signature) | Count/iter | A union (ms) | B union (ms) | Δ ms | Notes |
|---|---|---|---|---|---|
| <e.g. gemm_TNT → mha_seed> | … | … | … | … | <what runs in this window on each side> |
| ... | | | | | |

## Step 5: Communication (Volume + Exposed)

<!-- Skip this section if no NCCL kernels found. -->

### Volume

| Op | A count | A ms | A avg | B count | B ms | B avg | Δ ms |
|---|---|---|---|---|---|---|---|
| AllGather | … | … | … | … | … | … | … |
| ReduceScatter | … | … | … | … | … | … | … |
| AllReduce | … | … | … | … | … | … | … |
| **Total** | … | … | — | … | … | — | … |

### Exposed comm

| Profile | Comm kernel (ms) | Exposed (ms) | Hidden % |
|---|---|---|---|
| A | … | … | … |
| B | … | … | … |
| **Δ** | … | **… ms** | — |

**Reconciliation**: `exposed + non_nccl_union ≈ GPU_busy` from Step 2. A: <a> vs
<b> (within <c> ms). B: similar.

**Interpretation**: <is exposed comm a meaningful contributor, or is comm well-
overlapped on both sides?>

## Step 6: Root Causes

<!-- This section requires source paths for both profiles being compared.
If source is unavailable, replace this section with: "Step 6 requires source
access for both implementations. Skipped." -->

### Architectural picture (verified)

- **A**: <how is the iter structured — full graph capture? eager? what fuses what?>
- **B**: same.

### Root Cause 1 — <name>

| | A | B |
|---|---|---|
| Mechanism | … | … |
| Source evidence | <file:line> | <file:line> |
| Owned phenomena | <which Δ rows from Steps 2–5 this RC owns> |

**Net contribution**: <Δ ms>

### Root Cause N — <name>

<!-- repeat per RC, typically 3-6 total -->

### Δ ownership table

| Phenomenon (Steps 2–5) | Δ ms | Owning root cause(s) |
|---|---|---|
| Step 2 GPU idle | … | RCx |
| Step 3 anchor-time (GEMM+MHA) | … | RCy |
| Step 4 residual (op-group / window-work) | … | RCz |
| Step 5 exposed comm | … | RCw |
| **Sum** | **…** | **vs measured … (within … ms ✓)** |

### Recommendations (ordered by recoverable impact)

1. <Specific code/config change, file:line, expected delta>
2. ...

## Verification status

| Claim | Status |
|---|---|
| Step 1 anchor + per-iter time | Verified from sqlite |
| Step 2 GPU busy/idle | Verified from sqlite |
| Step 3 GEMM/MHA equality | Verified from sqlite + YAML |
| Step 4 breakdown | Verified from sqlite |
| Step 5 comm volume + exposed | Verified from sqlite |
| RC1 mechanism | <Verified from source / Inferred from trace / Unverified — empirical follow-up needed> |
| RC2 mechanism | … |
| RC<N> | … |

### Open follow-ups (not yet verified)

1. <Specific item, expected delta, how to verify>
2. ...
