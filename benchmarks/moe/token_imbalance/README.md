# MoE runtime token-imbalance benchmark

A reproducible benchmark for the MoE token-imbalance problem. One set of logical routing
workloads is executed by several **arms** (ways of running the same MoE layer), checked
against exact oracles, and measured per rank, together with the **rank-level load** each
workload puts on each EP rank.

Two drivers share the workloads, metrics and output format:

| Driver | What it runs |
|---|---|
| `benchmark.py` | the standard all-to-all dispatcher alone, in `dispatch_only`, `moe_fwd_bwd` or `training_step` mode, with exact dispatch/combine round-trip oracles |
| `compare_arms.py` | one real `MoELayer` per arm (all-to-all, HybridEP, MoonEP virtual experts, and the MoEScheduler planners when present), same parameters, inputs and replayed routes, rotated arm order |

## What is measured: rank-level load

For dropless MoE with EP, uneven expert routing turns into uneven work across EP ranks, and
the busiest rank sets the step time. Per-expert statistics alone cannot describe that: the
same hot experts can sit on one rank or be spread across several. Every arm therefore records,
per step, the destination load of every EP rank and three summaries of it
(`rank_load.py`):

| Quantity | Definition | Reads as |
|---|---|---|
| rank load `L_r` | token-expert assignments whose expert lives on EP rank `r`, after top-k expansion (contiguous expert placement, as in Megatron) | the full vector is kept |
| `maxvio` | `max_r L_r / mean_r L_r` | 1 when balanced; `maxvio - 1` is the busiest rank's relative overload |
| concentration | `1 - H(p) / log R`, `p_r = L_r / sum(L)` | 0 for a uniform load, 1 when one rank gets everything; separates loads with equal `maxvio`, e.g. `[40, 20, 20, 20, 0]` (0.172) vs `[40, 15, 15, 15, 15]` (0.065) |
| hot-rank persistence | share of consecutive-step transitions whose busiest rank is unchanged | 1 for a static hotspot, 0 when it moves every step |

Per-expert counts, max/mean and CV are still recorded alongside.

## Two experiments

They answer different questions and are reported in separate tables.

* **`trace_replay` (W\* workloads).** One fixed **logical** routing, defined from global
  expert ids alone, replayed at any EP layout. The rank-level imbalance it produces is a
  property of the layout; it is reported, not controlled.
* **`rank_target` (R\* workloads).** A routing **constructed for the given layout** so that
  every EP rank receives a chosen load exactly. This is where severity, concentration and
  temporal persistence are varied one at a time.

### trace_replay workloads

Seeded by `(workload, seed, step)` only; identical at every EP/ETP layout. Every row carries
exactly `topk` distinct experts. "Hot block" = the first `max(topk, E/4)` experts.

| ID | Construction | What it isolates |
|---|---|---|
| `W0_balanced` | round-robin distinct top-k | exact balance control |
| `W1_hot_expert_set` | 3/4 of the rows draw only from the hot block | per-expert heat, contiguous placement |
| `W3_spread_hot_experts` | W1 with expert ids relabeled so the hot block is strided across the expert range | **placement only**: per-expert counts are W1's exactly, as a multiset |
| `W4_random_balanced` | uniform random distinct top-k | sampling fluctuation around balance, vs W0's exact round-robin |
| `W5_zero_expert_rank` | the lower half of the experts is never selected | zero-load experts and ranks |
| `W6_rotating_hotspot` | W1's hot block moves by its own width every 3 steps | stale plans across steps (needs `--steps >= 4`) |
| `W7_burst` | 3 steady steps, 1 burst step, recovery, period 4 | a burst hidden in steady-state numbers (needs `--steps >= 5`) |

### rank_target workloads

Built for `--ep`; each rank's load is spread evenly over its local experts, so only the
rank-level shape varies. The realized per-rank load equals the target **exactly** and is a
hard oracle check. `--hot-maxvio` (default 2.0) sets the busiest rank's load as a multiple of
the mean.

| ID | Rank-load vector (EP=5, total 100, `--hot-maxvio 2`) | What it isolates |
|---|---|---|
| `R0_balanced` | `[20, 20, 20, 20, 20]` | balance control |
| `R1_hot_rank_spread` | `[40, 15, 15, 15, 15]` | severity: one hot rank, remaining load spread evenly |
| `R2_hot_rank_concentrated` | `[40, 20, 20, 20, 0]` | concentration: same busiest-rank load (identical `maxvio`), remainder on as few ranks as possible |

With two ranks the hot rank's share fixes the other one, so R1 and R2 coincide at EP=2;
concentration needs EP >= 3.

**Temporal persistence** is varied with `--schedule`, which only changes *which* rank carries
each load slot; the per-step load multiset, and therefore `maxvio` and concentration, stay the
same at every step:

| `--schedule` | Hot rank over steps | Needs |
|---|---|---|
| `static` (default) | always rank 0 | `--steps >= 1` |
| `rotate` | moves to the next rank every `--schedule-period` steps | `--steps >= period + 1` |
| `random` | a seeded random rank permutation per step | `--steps >= 2` |

A per-step workload or a non-static schedule **refuses to run** with fewer steps than it
needs, rather than replaying only its first step. Every measured iteration runs all
`--steps` steps in order, and each step is also timed on its own (`per_step_median_ms`).

## Modes

Every mode states what it includes and what it must not be used to claim.

| Mode | Includes | Must NOT be used to claim |
|---|---|---|
| `dispatch_only` | dispatcher preprocess/dispatch/postprocess, a stand-in local expert computation, and combine | GEMM cost, or a training-step result |
| `moe_fwd_bwd` | real `MoELayer` forward and backward; the router runs and its top-k is replaced by the workload through `RouterReplay` | full-model throughput or convergence |
| `training_step` | `moe_fwd_bwd` plus a real optimizer update | long-run convergence or full-size model performance |

In the model modes the routing map the layer actually used is captured and compared with the
workload at every step (`replayed_routing_exact`), so a model-mode number is always the
workload's number.

## What is inside the timed window

* **Included:** the dispatcher's four stages and the local expert computation; the router,
  backward pass and optimizer update in the model modes.
* **Excluded:** metric reduction, JSON writing, printing, and warmup iterations.

## Correctness oracle

Checked per rank, outside the timed window, with exact integer comparisons, at every step:

* every token row has exactly `topk` **distinct** experts;
* `sum_e assignments[e] == tokens * topk`, and the group histogram equals
  `tokens * topk * world`;
* dispatch -> identity expert -> combine restores every row exactly. The sentinel values are
  integers on a `2**-10` grid so bf16 represents them exactly, and the row sum is exactly
  `topk + rank`, which makes this an equality check rather than a tolerance check;
* the combine scaling factor is **measured** first and asserted to equal `topk`, so a change
  in the dispatcher's contract fails the run instead of silently rescaling the comparison;
* the dispatcher's receive splits are summed across ranks and reconciled against the global
  assignment count;
* `rank_target`: the realized per-rank load equals the target (`rank_target_exact`);
* model modes: the layer routed exactly the workload (`replayed_routing_exact`), the loss is
  finite and gradients are non-zero.

Two protocol facts this harness had to accommodate, both verified against the source:

1. `MoEAlltoAllTokenDispatcher.combine_*` **sums** expert contributions; the routing
   probabilities are applied inside the expert module
   (`experts.py: intermediate_parallel = intermediate_parallel * permuted_probs`). The harness
   therefore passes raw expert outputs to combine.
2. `dispatch_preprocess` consumes `probs` as a full `[tokens, num_experts]` matrix, not a
   compact `[tokens, topk]` one, and the argsort permute needs
   `num_experts * num_tokens >= tokens * topk`.

## Trace v1

With `--write-trace DIR`, every step of a workload is persisted as `DIR/<workload>/stepNNNN/`:

```
stepNNNN/
  manifest.json
  tensors/{token_ids,gates,source_rank,expert_ids}.pt
```

Validation on load (`trace_v1.load_trace`) rejects:

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
# trace_replay arms at EP=2, with every per-step workload exercised
torchrun --standalone --nproc_per_node=2 benchmark.py \
    --mode dispatch_only --ep 2 --topk 2 --tokens 128 --steps 6 \
    --workloads W0_balanced,W1_hot_expert_set,W3_spread_hot_experts,W6_rotating_hotspot,W7_burst \
    --out results/trace-ep2

# rank_target arms at EP=4: equal maxvio, different concentration, rotating hotspot
torchrun --standalone --nproc_per_node=4 benchmark.py \
    --mode moe_fwd_bwd --ep 4 --steps 4 --schedule rotate --schedule-period 1 \
    --workloads R0_balanced,R1_hot_rank_spread,R2_hot_rank_concentrated \
    --out results/target-ep4

# validation tests (CPU only)
pytest tests/unit_tests/transformer/moe/test_token_imbalance_*.py
```

Run `--help` for the full flag list. `--capability-json` writes the backend capability probe
result so a run records what the backend actually supported. `report.py RESULTS_ROOT` turns a
results tree into the per-experiment step-time and rank-level tables.

## Arms (`compare_arms.py`)

`arms.py` is the arm registry. Every arm builds the same layer from one shared configuration
(`arms.base_config`: BF16, TP=ETP=1, TE grouped experts, FP32 router, no aux loss, routing
replay on) and adds only the overrides that select its method; an arm that tries to change a
shared setting is rejected. New arms are added with `arms.register`.

| Arm | Method | Requires |
|---|---|---|
| `alltoall` | `MoEAlltoAllTokenDispatcher` | nothing beyond TE |
| `hybridep` | flex dispatcher, HybridEP backend | DeepEP `HybridEPBuffer` |
| `moonep` | `moe_virtual_expert_load_balance` (MoonEP virtual experts over HybridEP, #7450) | HybridEP with dense top-k routing (DeepEP@10d4dd7, as in `docker/Dockerfile.ci.dev`) |
| `scheduler_echo`, `scheduler_eplb`, `scheduler_moon_ep`, `scheduler_ultra_ep` | `moe_enable_scheduler` with that planner (#8500) | `TransformerConfig.moe_enable_scheduler` and the TE op fuser (not `--no-te-op-fuser`); UltraEP also needs the `ultra_ep` package |

All arms run with `use_transformer_engine_op_fuser=True` by default, as MoEScheduler requires
(`--no-te-op-fuser` turns it off for every arm alike).

MoonEP and the scheduler arms hand the experts a capacity-padded buffer (more rows than
`sum(tokens_per_expert)`). Only TE's native grouped-tensor GEMM consumes such a buffer, and TE
selects it only with cuBLASLt >= 13.3; with an older cuBLASLt it falls back to split-quantize and
the first expert GEMM fails with a `split_with_sizes` mismatch. `run.json` records the
cuBLASLt, TE, DeepEP, Triton and NCCL versions of each run.

Requirements are probed on the node (`arms.probe_requirements`). An unavailable arm is
listed in `run.json` with every unmet requirement and is not run; it is never replaced by
another arm. The scheduler arms use one replica slot per EP rank (`moon_ep`: one per logical
expert, as its planner requires) and `--scheduler-transport replica_nccl` by default.

Fairness rules the driver enforces:

* every arm gets the same initial logical parameters (copied from the first arm built), the
  same per-rank inputs and upstream gradients, and the same replayed routes, which are
  checked against the routes the layer executed at every call;
* `--warmup` unmeasured and `--iters` measured passes over all `--steps` steps, identical for
  every arm;
* arm order is rotated per workload (`--order rotate`), so no arm always runs first;
* numerics of every arm are compared with `--reference-arm` (default `hybridep`) on the first
  measured pass: relative L2 <= 0.01 and peak-normalized error <= 0.05 for the output, input,
  gate and parameter gradients (and updated parameters with `--learning-rate`). These are BF16
  tolerance gates, not bit-exactness claims.

The headline number is the worst-rank time per step (max over ranks of the CUDA-event time
for router, forward, backward and optional SGD), summarized in `summary.json` as median,
p10/p90 and min/max over `iters * steps` samples. Per-rank JSONL rows keep every sample,
logical expert and rank loads, hashes of inputs, routes and values, and the virtual-expert
plan.

```bash
torchrun --standalone --nproc_per_node=4 compare_arms.py \
    --arms alltoall,hybridep,moonep --steps 6 --warmup 2 --iters 5 --out results/arms-ep4
```

## Capability probe

`backend_adapter.py` reports what the all-to-all backend of `benchmark.py` supports **here**, from imports and measurement,
never from documentation. Unknown items are reported as `UNVERIFIED`. The expert computation
is a per-expert loop in every mode (`moe_grouped_gemm=False`), so absolute step times are
**not** comparable with a grouped-GEMM run, though comparisons between workloads at one
configuration remain valid.

## Metrics

Per arm, per rank: median/mean/p95/min step time and the per-step medians (CUDA events, ranks
aligned outside the measured interval), logical tokens/s, assignments/s, per-expert counts and
their max/mean and CV, and per step the destination rank loads, `maxvio`, concentration, and
the hot-rank persistence across steps; peak allocated and reserved bytes, and the full
correctness record. `recv_splits_per_source_rank` is the dispatcher's receive layout on this
rank (what it got from each source rank), which is not a rank-load measure.

An arm's headline number is the **worst rank** (max over ranks), because surfacing the
straggler is the point of a token-imbalance benchmark; averaging would hide it.

## Scope and known limitations

* **Every source rank sends the same routing block.** The workloads describe one source rank's
  rows and every rank replays them, so source-side (pre-dispatch) skew is not modeled; that is
  a sender-side dimension separate from the destination rank load measured here.
* **`benchmark.py` runs only the standard all-to-all backend**, with a per-expert loop
  (`moe_grouped_gemm=False`). Its times are not comparable with `compare_arms.py`, which uses
  TE grouped experts for every arm; the two are reported in separate tables.
* **`compare_arms.py` uses one NVLink host,** fixed per-rank source tokens and `topk`, and
  BF16. Recompute, CUDA graphs, zero source rows and other precisions are not exercised.
* **`strong` vs `weak` scaling is not a recorded field.** Increasing EP at fixed per-rank tokens
  increases the total token count, so results across EP widths are weak-scaling comparisons and
  must not be read as "EP=N is faster".
* Absolute latencies carry the noise of the node they were taken on. Use the paired-run
  protocol for comparisons; a single arm is not a result.
