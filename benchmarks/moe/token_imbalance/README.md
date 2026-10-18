# MoE runtime token-imbalance benchmark (standard all-to-all baseline)

A reproducible benchmark for the MoE token-imbalance problem: a **fixed logical routing
workload** is executed through the **existing standard all-to-all dispatcher**, checked
against an exact oracle, and measured per rank.

This is the first slice of the "public runtime token-imbalance benchmark" work item. It
deliberately does **not** implement ECHO / MoonEP / UltraEP or a runtime-aware scheduler.

## Modes

Every mode states what it includes and what it must not be used to claim.

| Mode | Includes | Must NOT be used to claim |
|---|---|---|
| `dispatch_only` | dispatcher preprocess/dispatch/postprocess, a stand-in local expert computation, and combine | GEMM cost, or a training-step result |
| `moe_fwd_bwd` | real `MoELayer` forward and backward, router included | full-model throughput or convergence |
| `training_step` | the above plus a real optimizer update | long-run convergence or full-size model performance |

## What is inside the timed window

* **Included:** the dispatcher's four stages, the local expert computation, and the backward
  pass in the model modes.
* **Excluded:** metric reduction, JSON writing, printing, and warmup iterations.
* The router **is** part of `training_step`/`moe_fwd_bwd` because it is part of a real step.
  In `dispatch_only` the routing is fixed input and the router is not executed.

## Workloads

Routing is determined by `(workload, seed, step)` alone, so the same logical workload is
reproduced regardless of execution method or EP/ETP layout. Every row carries exactly `topk`
distinct experts.

| ID | Construction | What it separates |
|---|---|---|
| `W0_balanced` | round-robin distinct top-k | fully balanced control (expert max/mean = 1.000) |
| `W1_hot_expert_set` | a fixed small expert set takes most selections | per-expert skew |
| `W2_hot_rank` | hot experts owned by one EP rank | per-rank skew, holding expert heat similar to W1 |
| `W3_spread_hot_experts` | same expert heat, hotspot spread across ranks | placement, not heat |
| `W4_source_ragged` | uneven per-rank source tokens | pre-dispatch skew |
| `W5_zero_expert_rank` | half the experts never selected | zero-load experts / ranks (expert max/mean = 2.000) |
| `W6_rotating_hotspot` | hotspot moves every 3 steps | stale plan/count across steps |
| `W7_burst` | 3 steady steps, 1 burst step, then recovery | hiding a re-plan in steady-state numbers |

Imbalance is configuration dependent: it grows as each rank holds fewer tokens, because a
fixed hot set then covers a larger share of a smaller local batch. Both measured points are
recorded rather than one being presented as "the" number.

**EP=4, 128 tokens per rank, topk 2** (group-global histogram):

| workload | expert max/mean | expert CV (pop.) |
|---|---|---|
| W0_balanced | 1.000 | 0.000 |
| W1_hot_expert_set | 3.312 | 1.309 |
| W2_hot_rank | 3.219 | 1.272 |
| W3_spread_hot_experts | 1.281 | 0.159 |
| W4_source_ragged | 1.281 | 0.141 |
| W5_zero_expert_rank | 2.156 | 1.007 |
| W6_rotating_hotspot | 3.312 | 1.319 |
| W7_burst | 1.219 | 0.126 |

**EP=4, 512 tokens per rank, topk 4**:

| workload | expert max/mean | expert CV (pop.) | rank max/mean |
|---|---|---|---|
| W0_balanced | 1.000 | 0.000 | 1.000 |
| W1_hot_expert_set | 1.781 | 0.744 | 1.000 |
| W2_hot_rank | 1.770 | 0.747 | 1.000 |
| W3_spread_hot_experts | 1.047 | 0.031 | 1.000 |
| W4_source_ragged | 1.086 | 0.046 | 1.000 |
| W5_zero_expert_rank | 2.000 | 1.000 | n/a |
| W6_rotating_hotspot | 1.770 | 0.742 | 1.000 |
| W7_burst | 1.066 | 0.040 | 1.000 |

`max/mean` is `max(counts) / mean(counts)`; the CV is the population CV. At the topk-4 point
W1/W2/W6 land in the same heat band (1.770-1.781), so the W1-vs-W2 difference there is
placement rather than heat.

## Measured results

Step time by workload at a fixed configuration, worst rank, median over 20 measured iterations
after 5 warmups. Only the logical routing changes down each row.

| configuration | W0 | W1 | W2 | W3 | W4 | W5 | W6 | W7 | spread |
|---|---|---|---|---|---|---|---|---|---|
| `dispatch_only` EP=1 k=2 T=128 | 1.346 | 1.275 | 1.436 | 1.191 | 1.305 | **1.108** | 1.297 | **1.313** | 1.296x |
| `dispatch_only` EP=2 k=2 T=128 | 1.806 | 3.606 | 4.714 | 1.956 | **4.866** | 4.746 | 4.427 | **1.831** | 2.694x |
| `dispatch_only` EP=4 k=2 T=128 | **4.710** | 3.292 | 1.568 | 1.589 | 1.579 | 2.570 | **4.791** | 1.611 | 3.056x |
| `dispatch_only` EP=2 k=4 T=512 | **4.941** | 1.644 | 1.751 | 2.624 | 1.730 | 2.686 | 2.763 | 1.782 | 3.007x |
| `dispatch_only` EP=2 k=2 T=2048 | 2.314 | 2.665 | 2.457 | 2.729 | 2.457 | **1.915** | 3.113 | **5.206** | 2.718x |
| `moe_fwd_bwd` EP=1 k=2 T=128 | 2.661 | 2.083 | 1.981 | 2.745 | 1.519 | **1.202** | 2.315 | **2.544** | 2.285x |
| `moe_fwd_bwd` EP=2 k=2 T=128 | 3.252 | 3.001 | 3.113 | 2.749 | 3.017 | **3.722** | 3.023 | **3.024** | 1.354x |
| `training_step` EP=1 k=2 T=64 | **13.099** | 14.118 | **15.566** | 9.166 | 9.352 | 9.083 | 9.737 | 9.278 | 1.714x |
| `training_step` EP=2 k=2 T=64 | **12.764** | 12.944 | 12.416 | 11.923 | **9.820** | 11.666 | 10.750 | 12.529 | 1.318x |

These are absolute times from one shared node, so they are not a cross-configuration
comparison: EP is varied at a fixed per-rank token count, which makes the total token count
grow with EP (weak scaling). Read a row, not a column.

Correctness: **72 of 72 arms across the nine configurations passed every oracle check**,
covering `assignments_match_T_times_k`, `combine_factor_is_topk`, `counts_len_ok`,
`distinct_topk_ok`, `global_histogram_exact`, `id_range_ok`, `local_counts_sum_ok`,
`recv_splits_assignments_ok`, `roundtrip_exact`, `roundtrip_matches_calibrated_factor`, and the
three `training_step` gates.

## Correctness oracle

Checked per rank, outside the timed window, with exact integer comparisons:

* every valid token row has exactly `topk` **distinct** experts;
* `sum_e assignments[e] == tokens * topk`, and the group histogram equals
  `tokens * topk * world`;
* dispatch -> identity expert -> combine restores every row exactly. The sentinel values are
  integers on a `2**-10` grid so bf16 represents them exactly, and the row sum is exactly
  `topk + rank`, which makes this an equality check rather than a tolerance check;
* the combine scaling factor is **measured** first and asserted to equal `topk`, so a change
  in the dispatcher's contract fails the run instead of silently rescaling the comparison;
* the dispatcher's receive splits are summed across ranks and reconciled against the global
  assignment count;
* `training_step` additionally requires a finite loss and non-zero gradients.

Two protocol facts this harness had to accommodate, both verified against the source:

1. `MoEAlltoAllTokenDispatcher.combine_*` **sums** expert contributions; the routing
   probabilities are applied inside the expert module
   (`experts.py: intermediate_parallel = intermediate_parallel * permuted_probs`). The harness
   therefore passes raw expert outputs to combine.
2. `dispatch_preprocess` consumes `probs` as a full `[tokens, num_experts]` matrix, not a
   compact `[tokens, topk]` one, and the argsort permute needs
   `num_experts * num_tokens >= tokens * topk`.

## Trace v1

A workload can be persisted and replayed:

```
<trace-dir>/
  manifest.json
  tensors/{token_ids,gates,source_rank,expert_ids}.pt
```

Validation on load rejects:

* an unsupported `schema_version`, a missing required field, or a missing tensor file;
* a tensor whose sha256 does not match the manifest;
* a shape/dtype mismatch against the manifest;
* a row whose top-k slots repeat an expert;
* NaN/Inf gates, negative gates, or gate rows that do not sum to 1;
* an out-of-range expert id or source rank;
* a trace with zero logical tokens.

Tensors are loaded with `weights_only=True`, so only plain tensors and primitive containers
are reconstructed. Nothing is silently truncated, reordered or padded: a validation failure
raises.

## Usage

```bash
# one arm
torchrun --standalone --nproc_per_node=2 benchmark.py \
    --mode dispatch_only --ep 2 --topk 2 --tokens 128 \
    --out results/arm1

# persist the workload
torchrun --standalone --nproc_per_node=2 benchmark.py \
    --mode dispatch_only --ep 2 --out results/arm1 --write-trace /abs/path/traces

# validation tests (CPU only)
python test_trace.py
```

Run `--help` for the full flag list. `--capability-json` writes the backend capability probe
result so a run records what the backend actually supported.

## Capability probe

`backend_adapter.py` reports what the backend supports **here**, from imports and measurement,
never from documentation. Unknown items are reported as `UNVERIFIED`. On a node without
`transformer_engine` it reports `grouped_gemm: UNSUPPORTED`, which means the expert computation
is a per-expert loop: absolute step times from such a run are **not** comparable with a
TE-enabled node, though baseline/patch comparisons on the same node remain valid.

## Metrics

Per arm, per rank: median/mean/p95/min step time (CUDA events, ranks aligned outside the
measured interval), logical tokens/s, assignments/s, expert and rank imbalance, raw
per-expert and per-rank counts, peak allocated and reserved bytes, and the full correctness
record.

An arm's headline number is the **worst rank** (max over ranks), because surfacing the
straggler is the point of a token-imbalance benchmark; averaging would hide it.

## Scope and known limitations

* **`dev` was not surveyed.** The MoE roadmap states that this work item is based on the `dev`
  branch. The node used for this work could not reach it, so this implementation is based on
  `main` (`d564dd01d`). Landing it may require layout changes.
* **Router replay is not wired in.** Routing is supplied at the dispatcher level, or produced
  by the real router in `training_step`. This is a separate path from the model's
  `RouterReplay`, and the two should not be treated as interchangeable.
* **Only the standard all-to-all backend exists.** There is no ECHO / MoonEP / UltraEP adapter;
  adding one should reuse that method's own code and configuration.
* **The grouped-GEMM path is untested** on nodes without `transformer_engine`.
* **`strong` vs `weak` scaling is not a recorded field.** Increasing EP at fixed per-rank tokens
  increases the total token count, so results across EP widths are weak-scaling comparisons and
  must not be read as "EP=N is faster".
* Absolute latencies carry the noise of the node they were taken on. Use the paired-run
  protocol for comparisons; a single arm is not a result.
