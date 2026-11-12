---
name: mcore-calibrate-dcp-runtime
description: Calibrate or evaluate cost coefficients for the DynamicUSP scheduler of the DCP runtime, either a table for the original cost profile or a measured profile for classic P2P such as the packaged H100 calibration. Use when preparing a calibration for the dynamic_usp scheduler type, refitting the H100 calibration, adding a measured profile for other hardware, or checking whether the cost terms match an execution configuration.
license: BSD-3-Clause
---

# Calibrate the DynamicUSP scheduler's cost model

Work relative to the Megatron-LM checkout. The reference for the runtime file,
the start-up checks and the trace records is
`megatron/core/datasets/dcp_runtime_scheduler/README.md`; launch arguments are
in the
[configuration reference](../mcore-configure-dcp-cost-model/references/model-options.md).

The `dynamic_usp` scheduler type has two cost profiles. The original profile
takes a coefficient table; the next three sections calibrate it. The packaged
H100 calibration is the result of the same procedure for an execution
structure the original equations do not describe;
[its section](#worked-example-h100-classic-p2p) says how it was measured, how
to refit it and how to add a profile for other hardware.

This workflow adapts DynamicUSP's hot-switch calibration method
(`skills/hot-switch-calibration/SKILL.md` at commit
`18eb0f53dbdb3a13b19e49c02fd1782178947c9a`).
The [upstream license](references/DYNAMIC_USP_LICENSE.txt) is retained. The
runtime ports the scalar profile and table from upstream commit
`1ff97a4ae0beb5e088fadb9e112265bdd194b24d`; it does not bundle the upstream
measurement scripts or Stage 2 optimizer, and the tree has no measuring or
fitting script of its own.

## Establish the model and measurement domain

Read `megatron/core/datasets/dcp_runtime_scheduler/models/usp_calib_v1.py` and
its table `usp_calib_v1.json` before choosing measurements. The original
profile fixes the upstream synthetic 44-layer shape, 24 query heads, head
dimension 160, U8 all-to-all plus ring, full-layer replay with FA cache and
four backward ring chunks. CP group sizes and DPxCP are 8, 16 or 32. A new
table changes the coefficients of those equations. It does not select another
attention architecture, layer layout, parallel layout or transport, and the
profile does not read the training model configuration.

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
fit the terms the profile reads. The table is JSON with these fields:

| Field | Content |
| --- | --- |
| `a2a.fits.U8_pre`, `a2a.fits.U8_post` | `floor_ms`, `t0_ms` and `bw_gbps` of the all-to-all before and after attention |
| `a2a.in_train_inflation_by_tier` | In-training inflation for `sp8`, `sp16` and `sp32` |
| `ring.fits.kv`, `ring.fits.dkv` | `t0_ms` and `bw_gbps` of the ring exchanges |
| `ring.beta_by_chunk_bytes` | Backward amplification by chunk size, keys such as `"298MB"` |
| `compute.fa_additive_law.fwd.P_tflops`, `compute.fa_additive_law.bwd.P_tflops` | Attention forward and backward throughput |
| `compute.moe_intrain_rate_tflops` | MoE throughput per rank |
| `compute.moe_rate_lut.tokens`, `compute.moe_rate_lut.tflops` | Token/rate curve below the knee: two lists of equal length, tokens increasing and ending at 16384 |
| `global.gamma_comp.pure_compute` | Compute multipliers `fwd` and `bwd` for `sp8_fa_cache`, `sp16_ring` and `sp32_ring`, fitted after subtracting exposed communication |
| `global.C_global_clean_ms` | Iteration tail, reconciled against real steps with profilers off |

All values are finite; throughputs, bandwidths, inflations and multipliers are
positive. A table without one of these fields is rejected while training
starts, with the field name. The upstream table has further explanatory
fields that the profile does not read.

Declare warm-up and measured counts and keep the observations of all ranks.
The upstream workflow recommends 20 measured steps per phase; use the user's
counts when given. Run collection, aggregation and fitting in the allocated
compute environment. Keep held-out errors, rank spread, instrumentation
overhead and control-run results separate from the fitting data. A job that
finishes does not show that the model fits.

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
SHA-256 of the file's bytes.

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

The measurement followed the upstream steps: fixed-layout anchor cases per CP
size with equal-load, low-load and sequence-count groups, 20 measured steps per
case, a refit after dropping two more steps, held-out cases declared before
collection, then real global batches replayed at every feasible microbatch
count to fit the overhead per microbatch and the iteration tail.

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
4. Check the measured domain in `check_domain` so that start-up names the
   first setting that differs, and record the domain and the fit errors in
   the coefficient file as the packaged one does.
