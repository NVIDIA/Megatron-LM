---
name: nsight-system-analysis
description: Analyze NVIDIA Nsight Systems (nsys) GPU profiles to find the per-iteration performance gap between two implementations of the same workload, or to break down a single profile. Use whenever the user provides one or two `.nsys-rep` / `.sqlite` traces and asks "why is X slower than Y?", "compare these two nsys profiles", "investigate this nsys trace", "find the perf gap", "analyze this GPU profile", "look at one forward pass / one decode step", "how many kernels per step", or anything similar. Also use when the user wants per-iteration time, GPU busy vs idle split, per-category kernel breakdown (GEMM/Conv/MHA/Norm), module-slicing between GEMM anchors, exposed communication time, single-forward-pass kernel-count/composition comparison, or source-level root-cause analysis of perf differences. Covers single-GPU and multi-GPU, training and inference. Do not use for non-GPU profiling, non-nsys profilers, or anything that isn't an nsys trace.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# Nsight Systems Performance Analysis

This skill produces rigorous per-iteration performance analysis of nsys traces — either comparing two profiles to find the gap, or breaking down a single profile.

## Three workflows

**Workflow A — comparative gap**: two `.sqlite` profiles of the same workload. Output: a structured report (`assets/report_template.md`) attributing the wall-clock per-iter Δ to specific kernels and (if source paths are provided) specific code changes.

**Workflow B — single-profile breakdown**: one `.sqlite` profile. Same template, but with absolute numbers instead of Δs and an "optimization priority" section instead of root causes.

**Workflow C — single forward pass (decode step)**: the fast, focused workflow for **inference** traces where only the steady-state decode loop matters and everything else (load, warmup, prefill, client-wait idle) is noise. One command isolates *one* forward pass and reports its wall time, kernel count, and per-category composition — for one profile, or side-by-side for two. **Prefer this over Workflow A/B when the user asks "look at one forward pass / one decode step", "how many kernels per step", "why is one step slower", or is comparing two inference engines (e.g. mcore vs vLLM).** See the dedicated section below.

The user's question tells you which workflow: "compare A and B" or "why is X slower" (training / whole-iter) → A. "Analyze this profile", "what's expensive in this run" → B. "One forward pass", "one decode step", "kernels per step", inference engine comparison → **C**.

## Inputs you should ensure you have

Before starting, confirm with the user:
1. **Profile path(s)** (one or two `.sqlite` or `.nsys-rep` files). If only `.nsys-rep` is provided, ask for or generate the `.sqlite` via `nsys export --type sqlite`.
2. **Approximate iteration count** (optional — the iter-anchor script will auto-detect, but a user-provided count makes warmup/cooldown trimming unambiguous).
3. **For Step 6 (root cause)**: source code paths or access for both implementations. Without source access, the analysis stops cleanly after Step 5 and reports phenomena only.

## The protocol

Follow Steps 1 → 5 (or 1 → 6 if source is available) **in order**. Every later step builds on the prior step's numbers; skipping or reordering breaks the attribution arithmetic. After each step, write that section of the report immediately — don't batch.

### Fast path: `scripts/run_all.py`

For routine analysis, run the integrated pipeline:

```
python scripts/run_all.py \
    --profile-a flat.sqlite [--profile-b nonflat.sqlite] \
    --yaml taxonomy.yml \
    --out out/
```

It executes Steps 1, 2, 3, 5 on each profile, decides Step 4 mode (op-group vs
module-slicing) from the auto-computed `fused_share_of_residual_pct`, then
runs the chosen Step 4 script. All intermediates (windows, busy/idle, categorize,
module-slice, exposed-comm) are written to `<out>/` as separate JSON files.
A consolidated `summary.json` plus arithmetic invariants are written and printed.

**Always start with the fast path.** Then read the per-step intermediate JSON
to fill the report template. The per-step sections below describe each script
in case you need to re-run an individual step with different flags.

## Workflow C: single forward pass (decode step) — `scripts/forward_pass.py`

Use this when the user cares about **one forward pass** of an inference engine and wants everything else treated as noise. It answers, in one command:

- What is the forward-pass (per-decode-step) wall time?
- How many kernels launch in one forward pass?
- Where does the GPU-time go, per category (attention / MoE GEMM / MoE routing / activation / norm / dense GEMM / comm / elementwise / …)?
- **When comparing two engines**: is any individual kernel slower on one side (`µs/kernel`), or does one side simply launch **more** kernels for the same work (`#`)? This is the crux for mcore-vs-vLLM-style gaps, where the slower engine is usually not running slower kernels — it's running ~3× as many small, unfused ones.

**Run (one profile):**

```
python scripts/forward_pass.py mcore.sqlite
```

**Run (compare two — label order preserved):**

```
python scripts/forward_pass.py mcore.sqlite vllm.sqlite --label-a mcore --label-b vllm
```

**How it isolates one pass (fully automatic):**
1. Picks the **busiest GPU/rank** (one `deviceId`) so ranks don't interleave.
2. Finds the **steady-state decode region** = densest contiguous 1 s-binned kernel window (drops load/warmup/prefill and client-wait idle).
3. Finds a **per-step anchor**: the kernel that fires ~once per forward pass (embedding / LM-head / sampling / metadata). Per-step kernels have the *fewest* launches among recurring kernels (per-layer kernels fire `n_layers×` more) and the lowest inter-arrival jitter. Its median inter-arrival period **is** the forward-pass wall time — cross-check it against TPOT.
4. Extracts **one representative step** (a typical anchor-to-anchor interval near the middle of the decode region) and reports launches, GPU-busy (interval union), idle, and the per-category breakdown.

**Overrides** (when auto-detection is wrong or the user already found the boundaries in the GUI):
- `--anchor <substring>` — force the once-per-step marker (e.g. `--anchor _fused_metadata_kernel` for mcore, `--anchor triton_red_fused__to_copy_embedding_rms_norm` for vLLM).
- `--step-window t0,t1` — force the window in **seconds** (Nsight GUI timeline time = `raw_ns / 1e9`), e.g. `--step-window 156.9498,156.9543`.
- `--device N` — force a specific `deviceId`.
- `--yaml taxonomy.yml` — replace the built-in inference taxonomy with the skill's regex taxonomy.
- `--json` — machine-readable output.

**Interpreting the comparison table:**
- `Δus > 0` on a row ⇒ that category costs more in profile A. Sort by Δ to get the ranked optimization targets.
- Same `µs/kernel` but higher `#` ⇒ the lever is **fusion / fewer launches**, not a faster kernel. (e.g. a separate SwiGLU activation kernel per layer, FC1 and FC2 as two grouped-GEMM launches, or a "storm" of tiny routing/permute kernels vs a single fused routing kernel.)
- Higher `µs/kernel` on the same shape ⇒ a genuine kernel-selection / tile / dtype finding — drill into it with Step 6.

**Caveats:**
- `GPU-time (Σ durations)` per category **sums** kernel durations and so over-counts wall time when streams overlap — that's intentional for composition. Use the reported **GPU-busy (interval union)** for the true single-step wall figure, and the anchor **period** for the forward-pass time.
- The built-in taxonomy is tuned for MoE decode; verify the category assignments once (as in Step 3's YAML verification) if a large "other/misc" bucket appears, and extend via `--yaml`.

For deeper attribution (source-level root cause, exposed comm, module-slicing) fall through to Steps 1–6 below, restricting the windows to the decode region this workflow identified.

### Step 1: Find a per-iter anchor and measure per-iter time

This is **always first**. Without a correct per-iter time, every later number is wrong.

**Run**: `python scripts/iter_anchor.py <sqlite> [--n-iters N] [--anchor PATTERN]` — emits the chosen anchor, detected iter count, the windows, and per-iter timing stats as JSON to stdout. **Drops iter 1 and iter N by default** (warmup + cooldown); pass `--keep-warmup-cooldown` to keep them.

The script auto-detects the anchor in priority order:
1. NCCL collectives (AllGather, ReduceScatter, AllReduce) — best for distributed training.
2. CUDA stream/device sync — good when present (often absent in graph-captured loops).
3. Optimizer step kernels (`AdamFunctor`, `AdamCapturable`) — training only.
4. HtoD memcpy — fallback; async/non-blocking, less precise.
5. Densest recurring kernel — final fallback for inference / fusion-graph workloads.

The script picks the candidate with constant integer N per iter and lowest within-iter jitter. **If detection is ambiguous (no candidate yields constant N), it exits with an error listing the candidates** — ask the user which to use, or ask for the iteration count.

**Cross-check**: if two anchors agree on per-iter time within ~1 ms, the anchor is reliable. The script reports this when possible.

**Drop warmup/cooldown**: drop iter 1 and iter N when reporting median. Report median + min + max over the remaining iters.

**Report**: per-iter median + range for each profile, absolute Δ and Δ%. This is the headline.

### Step 2: GPU busy vs GPU idle = CPU-bound time

GPU busy = wall time during which ≥1 stream is running a kernel or memcpy (interval union). GPU idle = `iter_time − busy`, and **is** the CPU-bound portion (host dispatch, Python overhead, sync wait).

**Run**: `python scripts/busy_idle.py <sqlite> --yaml taxonomy.yml --windows windows.json` — emits per-iter busy, idle, and per-stream union JSON. **Always pass `--yaml`** so the script can compute the non-NCCL stream variant (otherwise it warns and the non-NCCL fields are identical to the all-kernel fields).

Why this matters: it tells you whether the gap is GPU-side (kernel composition, fusion, comm) or CPU-side (launch latency, host code). Pursue the dominant axis in later steps; if both contribute, break each down.

**Include memcpys in the union.** Kernels alone undercount GPU busy when there's significant HtoD/DtoH activity.

**Per-stream union**: the longest single-stream union is the critical-path stream. Compute the non-NCCL variant too (NCCL on a co-located stream contaminates the metric — see references/pitfalls.md).

**Counterintuitive case to expect**: the *faster* profile may have the *longer* single-stream union. That happens when the faster profile concentrates compute on one main stream while the slower one fans the same work across many parallel streams. In that case the slower profile's longest-single-stream is shorter — but its total (union across all streams) is larger. This is a finding to note, not a contradiction; the wall-clock gap is measured by total iter time (Step 1), not by the longest single stream.

### Step 3: Heavy-compute categories — GEMM / Conv / MHA

Only these three are reported here. They are model-level ops that are essentially never fused into anything else (same matmul shape on both sides should produce the same nvjet/cutlass call). If they differ, that's a real algorithmic / tile-heuristic / kernel-selection finding.

**Norm goes in Step 4**, not here. Norm is commonly fused into neighbors (rope, AdaLN, gating); reporting a "norm gap" in Step 3 mistakes fusion-placement for an algorithmic finding.

**Run**: `python scripts/categorize.py <sqlite> --yaml taxonomy.yml --windows windows.json` — emits per-category time and a list of uncategorized kernels above a threshold.

**Verify the YAML before reporting numbers** (this is non-negotiable — see "YAML verification" below). The YAML at `references/taxonomy_template.yml` is a **starting point, not a golden reference**.

### Step 4: Per-op breakdown — op-group first, module-slicing as fallback

Default: **op-group** breakdown of the residual (everything outside GEMM / Conv / MHA / NCCL). Norm is included here. Categories are emitted by `categorize.py` using the full taxonomy from the YAML.

Op-group is more interpretable for vanilla workloads: rows are recognizable operators (rmsnorm, rope, elementwise, fp8_cast). Each row shows "how much time does this operator take on each side", which is directly actionable.

**Decision rule — when to switch to module-slicing**: op-group breaks down when the two implementations have different fusion patterns. Concrete check, automated by the script: compute the share of non-anchor compute time spent in **custom-fused** kernels (names matching `_*_fused_*`, names containing 2+ op-root tokens like `_qkv_split_norm_rope` or `_fused_ln_adaln`, or anything matching a YAML category named `*fused*`). If that share is **>10%** of non-anchor time in *either* profile, op-group attribution becomes unreliable (a single misallocated `add` or `clone` can swing rows by tens of ms — see references/pitfalls.md). Switch to module-slicing.

**Run for op-group (default)**: `python scripts/categorize.py <sqlite> --yaml taxonomy.yml --windows windows.json --residual-only` — same script, residual mode.

**Run for module-slicing (fallback)**: `python scripts/module_slice.py <sqlite> --yaml taxonomy.yml --windows windows.json --anchor-categories gemm,mha` — emits per-window times grouped by anchor-pair signature.

**For comparative module-slicing**, after running module_slice on both profiles, also run `python scripts/module_diff.py mod_a.json mod_b.json` to get a paired diff. The script handles the case where the same window signature has different per-iter counts on each side — it surfaces both the total Δ and a per-call Δ (normalized by min count) so you can see per-call cost differences regardless of count mismatch.

Report which mode was used and **why**. The decision rule's input (custom-fused share) goes in the report so the choice is auditable.

#### Module-slicing details (when used)

- Anchors are all GEMM + MHA kernels across **all compute streams** (NCCL excluded by name), ordered globally by start time.
- Anchor counts on both sides should be approximately equal (same model = same matmul count). If they differ significantly, that's a structural finding to report.
- Anchors may overlap (different streams): when `anchor[k+1].start < anchor[k].end`, the window has zero or negative width — skip and set the next window's left boundary to `max(anchor[k].end, anchor[k+1].end)`.
- Windows are grouped by `(left_anchor_category, right_anchor_category)` signature so the same logical region across N blocks aggregates into one row.
- Each window's time is the **interval-union** of non-NCCL work in that window, not the duration sum (sum overcounts parallel-stream overlap).
- Both sides are non-zero on every row by construction — that's the property op-group can't always give.

### Step 5: Communication — volume and exposed time

Skip if the profile has no NCCL kernels (single-GPU, inference, etc.). The script exits cleanly with `{"comm": "none"}` in that case.

**Run**: `python scripts/exposed_comm.py <sqlite> --yaml taxonomy.yml --windows windows.json` — emits per-collective counts and times, total comm time, exposed time (union of NCCL minus union of non-NCCL), and hidden percentage.

Two metrics:
- **Comm volume** (count × avg per collective type) — diagnoses bucketing / sharding differences.
- **Exposed comm time** — the wall-clock contribution. Most comm is hidden by overlap; only the exposed portion adds to iter time.

**Reconcile arithmetic**: `exposed_comm + non_NCCL_union ≈ GPU_busy` from Step 2 within ~0.5 ms. If not, you double-counted or mis-clipped — go back.

### Step 6: Root cause (only if source paths provided for all profiles being analyzed)

This is where the report becomes a finding rather than a measurement. Skip if source is unavailable; output a clear note that Step 6 needs source access.

**Coverage rule**: every Δ row in the final attribution table needs **all three** of:
1. **Source-level evidence on both sides** (file:line each, citing the actual code path that runs at runtime).
2. **Mechanism**: what specifically in the code causes the kernel selection / fusion / dispatch / scheduling to differ. Not "the heuristic picks differently" — *why* it picks differently.
3. **Actionable change**: the code or config change that would close the Δ, named at file:line.

If you can't produce all three for a given Δ, mark it "unverified — needs follow-up" rather than ship an unverified guess as a root cause.

**Drill-down templates** (use the one that matches each Δ):

- **GEMM/MHA tile or kernel-selection Δ**: list per-shape kernel-name + count + time. Diff. The same matmul shape with different kernel-name suffix means the same heuristic took different inputs — find which arg differs (epilogue, dtype, accumulate, alignment, fp8 recipe, layout flags). Read the actual call sites on both sides.
- **Op-group or module-slice window Δ**: dump the full ordered kernel list inside the top-N differing windows / rows on both sides. Identify the operator(s) each kernel implements. Trace back to source: is the difference a custom fused kernel on one side, a `torch.compile` boundary on the other, a framework-level decomposition, a precision/cast inserted by autocast?
- **Exposed comm Δ**: which collective is unhidden? What compute *should* have overlapped? Look for `cudaStreamWaitEvent`, sync-point insertion, bucket-size config (`bucket_size`, `align_param_gather`, `overlap_grad_reduce`).
- **GPU idle Δ**: count `cudaLaunchKernel` events on each side (CUPTI_ACTIVITY_KIND_RUNTIME). Diff. Identify what each extra launch is *for* (optimizer? grad clip? metric logging?). Look for CUDA-graph scope (capture region might be narrower on one side).

**Phenomenon vs cause traps** — see references/pitfalls.md:
- "The heuristic picks differently" is not a cause. Find the input that differs.
- "More launches" is not a cause. Find what the extra launches do.
- "Different bucketing" is not a cause. Find the config or env var that sets it.

**Collapse phenomena to fewer causes**: several Δ rows often trace to a single root cause (e.g. three different windows all slower because the same fused custom kernel is missing on one side). Final report should have 3–6 root causes, not 20 phenomena.

**Verification status table**: end the report with a table listing every claim and whether it's *verified from source*, *inferred from trace*, or *unverified — empirical follow-up needed*. This is required — without it, readers can't tell which findings are solid.

## Hard rules (apply throughout)

1. **Union, never sum, for wall time.** Multiple streams overlap; summing kernel durations overstates wall time 2–4×. Use interval union (`scripts/_lib.py:union_intervals`).
2. **YAML is a reference, not golden.** Always verify per-category matches by hand-inspecting the kernel names before reporting numbers. See "YAML verification" below.
3. **Apples-to-apples comparison.** In op-group mode, every row must be non-zero on both sides if the workload actually performs that op. In module-slicing mode, anchor counts must match (structural alignment).
4. **Phenomenon ≠ root cause.** Step 6 requires source-level evidence with file:line, mechanism, and actionable change.
5. **Every Δ must reconcile.** Final attribution table sum must match measured iter Δ to ≤0.5 ms. If it doesn't, find what you missed — don't round and ship.

## YAML verification (Step 3 prerequisite)

Before reporting any Step 3 numbers:

1. Run `categorize.py` and inspect the matched-kernel list per category. Confirm every matched kernel name actually belongs to its category. Common false positives: cuBLAS epilogue helpers caught by a loose `gemm` regex; unrelated kernels with "norm" in the name; Triton catch-all swallowing a fused norm or fused matmul.
2. Inspect the uncategorized list. For each name with non-trivial time (>1% of iter), decide: extend a regex to include it, or add a new category. Pay attention to fused-op kernels (`_qkv_split_norm_rope_kernel`, `_fused_ln_adaln_*` style) — they often need their own custom category.
3. Iterate: re-run, re-check, fix until matched-kernels-per-category-look-correct AND uncategorized-list-is-empty-of-significant-time. Save the iterated YAML as part of the deliverable.

For comparative mode, the YAML must produce verified matches in both profiles — kernels that exist in one profile may not exist in the other, but the regexes must be sound for whichever kernels do appear.

## Output: the report

Fill `assets/report_template.md`. Section structure is fixed (Steps 1–6 + Inputs + verification-status table). Numbers go in the tables; prose goes between them.

Two arithmetic invariants the final report must satisfy:
- **Step 2 sanity**: `iter_time = GPU_busy + GPU_idle` on each side, within ~0.2 ms.
- **Step 6 sanity** (comparative mode): `Σ(Δ rows in attribution table) = measured iter Δ`, within ~0.5 ms.

If either invariant fails, the analysis is incomplete or wrong — fix before finalizing.

**Common arithmetic trap when attributing the gap**: GPU busy is an *interval union*, but Step 4 window-work is sometimes a *sum* across parallel streams. When non-anchor kernels run concurrently on multiple compute streams, the sum overcounts wall-time by the overlap. Always use the **union** column for attribution (`module_slice.py` emits both `*_union_*` and `*_sum_*`; use union). If you find a leftover ~ms-scale residual between `Σ(Δ rows)` and measured iter Δ that you can't account for, this is usually the culprit — recompute with unions throughout.

**A second small (~1–3 ms) residual is expected** between (Step 4 anchor + Step 4 window-union + Step 5 exposed-comm) and (Step 2 GPU-busy) when NCCL kernels are co-located on the same streams as compute. Step 4 excludes NCCL by name from windows, but if a tiny non-NCCL kernel happens to be straddled by NCCL clipping it can slip out of the accounting. Flag this in the report; do not paper over it.

## Two-page references

- `references/taxonomy_template.yml` — minimal generic regex taxonomy. Copy and extend per workload.
- `references/sql_recipes.md` — paste-ready SQL queries for ad-hoc inspection.
- `references/pitfalls.md` — common traps and rejected approaches (sum-overcounting, single-stream anchor, op-group with heavy fusion, single-rank assumption under TP/PP, YAML-as-golden, "the heuristic flipped").
