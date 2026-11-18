---
name: mcore-calibrate-dcp-runtime
description: Calibrate or evaluate cost coefficients for the DynamicUSP scheduler of the DCP runtime, either a table for the original cost profile or a measured profile for classic P2P such as the packaged H100 calibration. Use when preparing a calibration for the dynamic_usp scheduler type, fitting its sample weight, tier factors or step terms, switching on Stage 2, refitting the H100 calibration, adding a measured profile for other hardware, deciding what to measure again after the tensor-, expert- or pipeline-parallel size changed, or checking whether the cost terms match an execution configuration.
license: BSD-3-Clause
---

# Calibrate the DynamicUSP scheduler's cost model

Work relative to the Megatron-LM checkout. The reference for the runtime file,
the start-up checks and the trace records is
`megatron/core/datasets/dcp_runtime_scheduler/README.md`; launch arguments are
in the
[configuration reference](../mcore-configure-dcp-cost-model/references/model-options.md).

The `dynamic_usp` scheduler type has two cost profiles. The original profile
takes a coefficient table; the sections up to
[Validate and use the table](#validate-and-use-the-table) calibrate it. The
packaged H100 calibration is the result of the same procedure for an execution
structure the original equations do not describe;
[its section](#worked-example-h100-classic-p2p) says how it was measured, how
to refit it and how to add a profile for other hardware.

This workflow adapts DynamicUSP's hot-switch calibration method
(`skills/hot-switch-calibration/SKILL.md` at commit
`18eb0f53dbdb3a13b19e49c02fd1782178947c9a`).
The [upstream license](references/DYNAMIC_USP_LICENSE.txt) is retained.

## What the runtime ports

From upstream commit `1ff97a4ae0beb5e088fadb9e112265bdd194b24d` the runtime
ports `solve_gbs`: the Stage 1 allocation, the search over microbatch counts
and the step-time objective that scores each count, for one and for more than
one pipeline stage, with the Stage 1 sample weight, the scalar profile and its
table, and Stage 2 (off by default). A run with more than one pipeline stage
and no virtual stages is scored with Megatron's order of a training step,
which is not upstream's. Upstream describes the measurements in
its calibration skill and ships no measuring or fitting script; this tree has
none either.

The upstream procedure calibrates three models. Each result has this place in
the coefficient file of the original profile:

| Upstream model | Result | Place |
| --- | --- | --- |
| 1, sample weight | A, B, C | `solver.sample_weight`; see [Sample weight](#sample-weight) |
| 2, group time | Rates, LUT, multipliers, all-to-all and ring curves, inflations, ring amplification, `C_global` | The table fields in [Collect and fit the terms](#collect-and-fit-the-terms) |
| 2, forward-KV exposed fraction | A fixed exposed fraction, or the rho LUT | None. The profile uses 0.87 for a ring of four (CP 32) and 0 for a ring of two (CP 16), and does not read the rho LUT. |
| Derived from 2 | Tier factors f(s) | `solver.tier_time_factors`; see [Tier factors](#tier-factors) |
| 3, step time | Overhead per microbatch; forward shares | `solver.pipeline`; see [Step terms](#step-terms) |

Stage 2 has no fit of its own: the profile computes its sample loads from the
group-time fields; see [Stage 2](#stage-2).

## Establish the model and measurement domain

Read `megatron/core/datasets/dcp_runtime_scheduler/models/usp_calib_v1.py` and
its table `usp_calib_v1.json` before choosing measurements. The original
profile fixes the upstream synthetic 44-layer shape, 24 query heads, head
dimension 160, U8 all-to-all plus ring, full-layer replay with FA cache, four
backward ring chunks and the forward-KV exposed fraction. CP group sizes and
DPxCP are 8, 16 or 32. A new table changes the coefficients of those
equations. It does not select another attention architecture, layer layout,
parallel layout or transport, and the profile does not read the training model
configuration.

Record the actual model and layers, the tensor, expert and pipeline parallel
sizes, precision, recomputation, attention and communication implementations,
physical rank placement and hardware, and compare them with every cost term
the profile uses. P2P execution needs its own feature map and communication
measurements; renaming an all-to-all table does not provide them. If the fixed
equations do not match the requested configuration, identify the required
model extension before collecting a long run, as the worked example does for
classic P2P. Keep the user's requested model scale and communication mode.

The packaged table is the upstream example table and was not fitted to a
Megatron model. Keep its bytes and attribution unchanged and write a new table
as a separate file.

## Collect and fit the terms

Pin source, environment, measurement boundaries and case lists. Define fitting
and held-out cases across logical length, sample count and the CP sizes of the
profile. Record padding and physical ownership, not only logical token counts.
Use the production execution settings and verify the communication
implementation in use before measuring.

Collect offline communication and compute sweeps and in-training anchors, then
fit the terms the profile reads. An anchor is a training run at
pipeline-parallel size 1 with one microbatch per step in which every CP group
has one fixed size and holds fixed samples of one length; run it per CP size
with several lengths and with groups of one and of several samples. Record,
per step and rank, the forward and backward time of every layer, the
communication waits and the step time, with the group's sample lengths.

The table is JSON with these fields:

| Field | Content |
| --- | --- |
| `version` | `"v1.1"`, the schema version of the upstream table. A table with another value, or without the field, is rejected. |
| `a2a.fits.U8_pre`, `a2a.fits.U8_post` | `floor_ms`, `t0_ms` and `bw_gbps` of the all-to-all before and after attention |
| `a2a.in_train_inflation_by_tier` | In-training inflation for `sp8`, `sp16` and `sp32` |
| `ring.fits.kv`, `ring.fits.dkv` | `t0_ms` and `bw_gbps` of the ring exchanges |
| `ring.beta_by_chunk_bytes` | Backward amplification by chunk size, keys such as `"298MB"` |
| `compute.fa_additive_law.fwd.P_tflops`, `compute.fa_additive_law.bwd.P_tflops` | Attention forward and backward throughput |
| `compute.moe_intrain_rate_tflops` | MoE throughput per rank |
| `compute.moe_rate_lut.tokens`, `compute.moe_rate_lut.tflops` | Token/rate curve below the knee: two non-empty lists of equal length, tokens strictly increasing |
| `global.gamma_comp.pure_compute` | Compute multipliers `fwd` and `bwd` for `sp8_fa_cache`, `sp16_ring` and `sp32_ring`, fitted after subtracting exposed communication |
| `global.C_global_clean_ms` | Iteration tail, reconciled against real steps with profilers off. It is added to every predicted step time and changes no plan; with Stage 2 on it only widens the tie tolerance of a re-pack. |
| `solver` | Optional: the coefficients of the search, described in the next four sections. Another key inside it is rejected. |
| `compute.in_train_overhead_factors.OF_fa_fwd`, `ring.in_train_bw_factor` | Read only by the ring guard of Stage 2: required when `solver.stage2.enabled` is true and the run forms a CP size above 8, without effect otherwise; see [Stage 2](#stage-2). Upstream requires them in every table. |
| `ring.fwdkv_exposure_frac_by_rho`, `c_seg_ms` under `compute.fa_additive_law` | Fields of the upstream table that the profile does not read. A refit may fill them; they have no effect here. Upstream requires them, so keep them in a table that upstream should also load. |

All values are finite: NaN or Infinity anywhere in the file is rejected when
it is read. Throughputs, bandwidths, inflations and multipliers are positive.
A table without one of the fields above the `solver` row is rejected while
training starts, with the field name.

Declare warm-up and measured counts and keep the observations of all ranks.
The upstream workflow recommends 20 measured steps per phase; use the user's
counts when given. Run collection, aggregation and fitting in the allocated
compute environment. Keep held-out errors, rank spread, instrumentation
overhead and control-run results separate from the fitting data. A job that
finishes does not show that the model fits.

## Sample weight

Stage 1 gives every sample one weight and balances the weight per rank. For a
sample of L tokens the weight is

```text
w(L) = max(1, A * L^2 / 1e10 + B * L / 1e5 + C)
```

Fit A, B and C by least squares over all anchor points of the smallest CP
size, on the forward time of the whole model on one rank:

```text
block forward time = A * sum(L_i^2) / 1e10 + B * sum(L_i) / 1e5 + C * n
```

where the L_i are the sample lengths of the group and n is their number. The
anchors need several lengths, or A and B cannot be separated, and groups with
different numbers of samples, or C cannot be separated. Only relative weights
matter. Write the result into `solver.sample_weight` of the table; see
[The solver object](#the-solver-object).

The default is A = 7706.03421944, B = 6528.86747592, C = 351.14154099:
upstream's published measured fit, which is not derived from the packaged
table and does not describe your model. A table without
`solver.sample_weight` uses it: the start-up line then has a notice that the
default A, B, C are in use, and the trace shows `"default": true` in
`statistics.stage1_weight`.

## Tier factors

With `solver.tier_time_factors` Stage 1 balances predicted time instead of
weight. The factor f(s) of a CP size s is the time per unit of per-rank weight
in a group of s ranks, relative to the smallest size. It is derived from the
fitted group-time model and not fitted separately:

```python
from megatron.core.datasets.dcp_runtime_scheduler.models.usp_calib_v1 import USPCalibV1CostModel
from megatron.core.datasets.dcp_runtime_scheduler.runtime.planner import RuntimePlanGeometry

geometry = RuntimePlanGeometry(
    world_size=32,  # DPxCP
    tokens_per_rank=32768,  # --max-seqlen-per-dp-cp-rank
    ep_size=1,  # expert-parallel size
    allowed_cp_sizes=(8, 16, 32),  # the CP sizes the run forms
    dataset_alignment=64,  # 2 x DPxCP
    max_group_seqs=64,  # sequences per CP group
    align_group_starts=True,
)
model = USPCalibV1CostModel(
    geometry, allow_provisional=True, calibration="calibration/calib_table_v1.json"
)
reference_tokens = 216358  # p99 sample length of your data
factors = model.derive_tier_time_factors(reference_tokens)
```

`derive_tier_time_factors` prices one sample of `reference_tokens` tokens
alone in a group of every CP size (compute, all-to-all and ring, without the
overhead per microbatch) and returns the group time times the size, relative
to the smallest size. Use the p99 sample length of the production data as the
reference: short samples have a much larger tier penalty and would
over-correct. For the packaged table a reference of 216358 tokens gives 1.0,
1.105 and 1.278 for CP 8, 16 and 32, and a reference of 8192 tokens gives 1.0,
1.424 and 2.649. Write the table into `solver.tier_time_factors`, with the CP
sizes as string keys in plain decimal digits; see
[The solver object](#the-solver-object).

- Derive the table again after every refit of the group-time coefficients. A
  stored table does not follow the coefficients it was derived from.
- Values are at least 1; a table with a smaller value is rejected. A table of
  ones, or no table, keeps weight balance.
- Keys are CP sizes of the profile. A key above the largest size the run
  forms is ignored, so one table serves runs with DPxCP 8, 16 and 32 that
  have `--min-dynamic-context-parallel-size 8`. A run at DPxCP 32 with a
  minimum size of 16 needs a table derived with the `allowed_cp_sizes` of
  that run, (16, 32): upstream rejects a key the run cannot form, the runtime
  accepts it and does not rescale the factors.
- The factors are meant for a profile in which the time of a sample per rank
  falls when it is spread over more ranks, so that f(s) / s falls with s, as
  in the original profile. For a fixed-buffer classic-P2P recipe such as the
  H100 calibration, f(s) / s grows with s: a sample whose predicted time
  exceeds the per-rank mean at the smallest size that holds it fits at no
  larger size either, and Stage 1 gives it the widest group.

## Step terms

The step time that scores a microbatch count is, for the original profile at
pipeline-parallel size 1: the microbatches run one after the other with the
group times of the table, only the ranks of one CP group wait for each other,
the step ends when the last rank finishes,
`solver.pipeline.per_mb_overhead_ms` is added for every microbatch and
`global.C_global_clean_ms` once. At a pipeline-parallel size P above 1 with V
virtual stages, the interleaved schedule is replayed instead, over P x V model
chunks of equal time, on the forward and backward time of every group; without
virtual stages, Megatron's order over P chunks, in which stage r first runs
min(P - r - 1, n) warm-up forward microbatches of the n of a step.

The upstream procedure fits three step terms in a run with pipeline
parallelism:

- The forward shares `solver.pipeline.compute_fwd_share` and
  `solver.pipeline.comm_fwd_share` are the forward part of a group's compute
  time and of its all-to-all plus ring time. At pipeline-parallel size 1 they
  change no prediction and are checked and recorded only.
- The overhead per microbatch is fitted as upstream: take 3 to 5 real global
  batches and run every candidate microbatch count of each. If every step
  uses the same count, the overhead cannot be told apart from the fixed cost
  per step.

The planner returns the layout of every candidate count with the Python
option `keep_candidate_layouts` (it is not a key of the runtime file); for
more than one pipeline stage also pass `pipeline=(P, V)`, and for a run
without virtual stages `pipeline=(P, 1), pipeline_schedule="non_interleaved"`
(`pipeline=(P, 1)` alone replays upstream's order for one chunk per stage):

```python
from megatron.core.datasets.dcp_runtime_scheduler.runtime.dynamic_usp import DynamicUSPRuntimePlanner

# {sample id: logical length} of one global batch
lengths = {0: 481280, 1: 811008, 2: 30720, 3: 91136, 4: 6144, 5: 154624}
planner = DynamicUSPRuntimePlanner(geometry, model, keep_candidate_layouts=True)
planner.plan(lengths)
for candidate in planner.last_statistics["candidates"]:
    print(candidate["microbatches"], candidate.get("predicted_ms"), candidate.get("layout"))
```

A candidate with a layout lists, per microbatch, the first rank, the CP size
and the sample ids of every group. A candidate without one has a `rejection`
instead. Replay each layout as a fixed layout, measure the step time and
subtract the prediction made with an overhead of 0. Replaying a layout needs
a data path that feeds the listed groups unchanged; the tree has none, so
this step uses your own harness. Fit the residual linearly in the microbatch
count with one intercept per batch; the slope is
`solver.pipeline.per_mb_overhead_ms`. The residual should be linear in the
count.

## Stage 2

Stage 2 re-packs the samples of each microbatch on its CP groups; the README
section "Stage 2" has its rules. Only the original profile offers it,
`"stage2": {"enabled": true}` in [the solver object](#the-solver-object)
switches it on, and it has no fit of its own:

- The work of a sample and its load in a group come from the group-time fields
  (the compute time at CP 8 and `global.gamma_comp`), and
  `solver.stage2.keep_threshold` (default 8.4, upstream's) is a rule, not a
  measurement.
- With a CP size above 8, upstream's ring placement guard reads two more table
  fields, and training stops while it starts when one is missing.
  `compute.in_train_overhead_factors.OF_fa_fwd` is a non-empty list of
  positive numbers: the forward attention time in training relative to
  `compute.fa_additive_law.fwd` (the guard takes the smallest).
  `ring.in_train_bw_factor` is a positive number: the ring bandwidth in
  training relative to `ring.fits.kv`. Upstream gives both as measured in a
  training window, without a procedure; measure them again after a change of
  TP or EP.
- The guard keeps a re-pack only when the mean segment of each group of s
  ranks above 8, sum(L^2) / sum(L) over its sample lengths L, is at least 0.4
  x (s / 8) x 1000 x `compute.fa_additive_law.fwd.P_tflops` / min(`OF_fa_fwd`)
  / (`ring.fits.kv.bw_gbps` x `ring.in_train_bw_factor`) tokens: 33368 at CP
  16 and 66736 at CP 32 for the packaged table.

## The solver object

The results of the last four sections go into one `solver` object at the top
level of the table:

```json
"solver": {
  "sample_weight": {"A": 7706.03421944, "B": 6528.86747592, "C": 351.14154099},
  "tier_time_factors": {"8": 1.0, "16": 1.105, "32": 1.278},
  "pipeline": {
    "per_mb_overhead_ms": 0.0, "compute_fwd_share": 0.25, "comm_fwd_share": 0.5,
    "fitted_at": {"pipeline_parallel": 2, "virtual_stages": 2, "schedule": "interleaved"}
  },
  "stage2": {"enabled": false, "keep_threshold": 8.4}
}
```

The table has one `solver` object; every part is optional, and another key or
a key given twice is rejected. The numbers shown are the default A, B, C, the
factors of the packaged table at a reference of 216358 tokens, example step
terms and the Stage 2 defaults. `fitted_at` records the run in which the
`pipeline` terms were fitted and changes no prediction; it takes all three
keys: the pipeline-parallel size, the virtual stages per pipeline stage (1
without virtual stages) and the schedule, `"interleaved"` with virtual stages
and `"non_interleaved"` without.

## After changing TP or EP

A coefficient file describes the tensor- and expert-parallel sizes it was
measured with. After changing either size:

- Keep the solver code and the file format; they do not depend on the sizes.
- Offline ring curves (`ring.fits`) can be reused for the same hardware and
  communication implementation.
- Repeat the anchor runs and refit everything that is measured in training:
  the `compute` fields, `global.gamma_comp`, the `a2a` fields,
  `ring.beta_by_chunk_bytes`, `global.C_global_clean_ms`,
  `solver.sample_weight` and the `solver.pipeline` terms. Then derive
  `solver.tier_time_factors` again.

The original profile reads neither size, so it does not notice the change: a
run that passes the two start-up checks of the runtime scheduler (sequence
parallelism needs tensor-parallel size 1; the expert-parallel size must divide
DPxCP) plans with the old numbers without a message. The H100 calibration
states tensor-parallel size 1 and expert-parallel size 8 in its domain:
start-up rejects a run with other sizes, and the loader rejects a refit file
that states other sizes, so such a run needs
[its own measured profile](#add-a-measured-profile-for-other-hardware).

After changing the pipeline-parallel size or the virtual stages, fit the three
`solver.pipeline` terms again and update `fitted_at`; the group-time fields do
not depend on them. With more than one pipeline stage the start-up line says
when `fitted_at` names another run; at pipeline-parallel size 1 it is not
compared, and a `per_mb_overhead_ms` fitted elsewhere is used without a
message.

## Validate and use the table

Compare the predictions of the `dynamic_usp` configuration with held-out
observations at the same logical lengths, physical padding, CP groups and
timeline. Check communication exposure and full-step closure separately. The
profile includes compute, MoE, communication and the iteration tail; it takes
no `non_attention` input.

Keep the table as a separate file with its measured domain, source identity,
raw-data digests and fit report, and name it in the runtime file:

```json
{
  "type": "dynamic_usp",
  "calibration": "calibration/calib_table_v1.json"
}
```

Launch with `--dynamic-cp-runtime-config /path/to/runtime.json`. A relative
`calibration` path is resolved beside that file. Without `calibration` the
packaged table is used. The file must not change during a run.

To confirm which file a run uses, set `"trace_dir"` in the runtime file and
read `model_metadata.calibration_path` and `model_metadata.calibration_sha256`
in any `consume` record of `<trace_dir>/rank<NNNNN>.jsonl`; the digest is the
SHA-256 of the file's bytes. `model_metadata.solver` in the same record shows
the solver coefficients in use.

Report the table and its digest, the measured configuration and transport,
held-out module and step errors, the result of the start-up validation and the
remaining coverage limits.

## Worked example: H100 classic P2P

The package ships one result of this procedure:
`megatron/core/datasets/dcp_runtime_scheduler/models/qwen3_h100_p2p.json`,
loaded by `qwen3_h100_p2p.py` and selected with

```json
{
  "type": "dynamic_usp",
  "hardware": "h100",
  "max_cp_size": 8
}
```

It was measured for full Qwen3-30B-A3B on 32 H100 with classic P2P context
parallelism. The [measurement record](references/qwen3-h100-p2p.md) has the
recipe, the measurements, the fit results and the content of the file; the
README lists the settings that start-up checks.

The example shows the step that precedes any measurement: decide whether the
original equations describe the execution. They model U8 all-to-all plus ring
with groups of 8, 16 or 32 ranks. Classic P2P issues no all-to-all and forms
ring groups of 1, 2, 4 or 8 ranks, and with a fixed token buffer per rank most
of the per-layer time is a constant of the CP size, for which the upstream
table has no entry. The upstream procedure asks for a profile of your own in
that case. The result is a second cost profile of the `dynamic_usp` type with
its own terms per CP size, planned by the same DynamicUSP search; it is not a
`calib_table_v1.json`.

The measurement followed the upstream steps for the group-time model and the
step terms: fixed-layout anchor cases per CP size with equal-load, low-load
and sequence-count groups, 20 measured steps per case, a refit after dropping
two more steps, held-out cases declared before collection, then real global
batches replayed at every feasible microbatch count to fit the overhead per
microbatch and the iteration tail.

Model 1 of the procedure, the Stage 1 sample weight A, B, C, was not fitted,
and no tier factors were derived; the scheduler uses a weight of its own, see
[What was not fitted](references/qwen3-h100-p2p.md#what-was-not-fitted). A
refit may add `solver.sample_weight`, fitted as in
[Sample weight](#sample-weight); the scheduler then uses the A, B, C form.
This calibration offers neither Stage 2 nor more than one pipeline stage.

### Refit the same recipe

Write a file with the content listed in the
[measurement record](references/qwen3-h100-p2p.md#content-of-the-coefficient-file)
and name it with `calibration` next to `"hardware": "h100"`. The loader
accepts this recipe only: the file must list the same `domain.config` settings
as the packaged file and state the global batch size of its step fit, and its
`domain` is checked against the training configuration while training starts.
Confirm the file in use through `trace_dir` as described above.

### Add a measured profile for other hardware

A different model, hardware or parallel layout needs its own measurement and
its own cost model:

1. Add the cost model under
   `megatron/core/datasets/dcp_runtime_scheduler/models/`. It implements the
   `DcpCostModel` protocol of `models/base.py`; `qwen3_h100_p2p.py` is the
   example.
2. Register its hardware name in `_MEASURED_DYNAMIC_USP` and its class in
   `model_class`, both in `models/__init__.py`. The runtime file then selects
   it with `"hardware": "<name>"`. A new hardware name also gets its GPUs per
   NVLink domain in `NVLINK_DOMAIN_SIZES` in `models/base.py`; the cost model
   takes `nvlink_domain_size` from that table, as `qwen3_h100_p2p.py` does,
   and the measurement places ranks consecutively, domain by domain.
3. Declare on the class what the scheduler must respect:
   `largest_cp_size`, the largest CP group the model prices (the scheduler
   forms no larger group); `pack_limits(cp)` returning a `PackLimits` with
   `max_sample_length` or `max_sequences` when the model prices only shorter
   samples or fewer sequences than the scheduler would otherwise place in a
   group; `required_global_batch_size` when its step terms describe one
   global batch size (start-up and the planner then reject any other).
4. Give the scheduler its Stage 1 coefficients: `parse_solver_section` in
   `models/solver_section.py` reads the `solver` section of the coefficient
   file; set `sample_weight_coefficients` to its `(A, B, C)` and
   `tier_time_factors` to its table, as `qwen3_h100_p2p.py` does. Without the
   first the scheduler falls back to the weight described in
   [What was not fitted](references/qwen3-h100-p2p.md#what-was-not-fitted);
   without the second a table in the file has no effect. Unless the model
   implements the Stage 2 members that `DcpCostModel` in `models/base.py`
   lists, pass `stage2_offered=False` as `qwen3_h100_p2p.py` does: otherwise
   `solver.stage2` in a file is accepted without effect. In `timing()`,
   declare `first_forward_dp_cp_sync: False` when only the ranks of one CP
   group wait for each other, as in DynamicUSP's objective; without it all
   ranks are synchronised after the forward pass of the first microbatch.
5. Check the measured domain in `check_domain` so that start-up names the
   first setting that differs, and record the domain and the fit errors in
   the coefficient file as the packaged one does.
