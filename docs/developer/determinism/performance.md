---
orphan: true
---

# Determinism performance measurements

`perf_breakdown.sh` gates on repeated **unprofiled** training timings. Nsight is
an optional separate diagnostic. JSON and Markdown artifacts retain all paired
runs, per-step samples, revisions, GPU identifiers, driver/package versions,
and effective mode settings. A timing pass does not establish determinism.

## Protocol

The default is three deterministic/default pairs, each with 20 warmup steps and
50 measured steps. Every arm starts a fresh process; pair order alternates to
reduce order bias. Both arms use identical model dimensions, input seed,
parallelism, and Python dependencies. Determinism-owned environment variables
are explicitly reset before launch, including in the default arm.

The report uses the median iteration time of each run, then ratios within each
pair. A paired bootstrap interval describes run-to-run uncertainty. Fewer than
three pairs or an interval that straddles the limit produces `inconclusive`
(exit 2), not a pass. A confirmed regression or invalid/incomplete timing log
exits 1. A valid pass exits 0. Three pairs are an initial CI budget, not a
guarantee that rare noise is characterized; increase the count when calibrating.

The existing 1.35 deterministic/default limit is retained. The optional
base-to-head comparison defaults to 1.05. Thresholds are configurable through
the Python CLI and should be calibrated on the target recipe and hardware.

The current metric is Megatron's logged iteration duration on its logging rank.
The reader requires a single timing stream and every expected step, and rejects
duplicate/restarted iterations, missing samples, nonfinite values, and zeros.
It does not introduce barriers or synchronization inside the training schedule.

## Run a recipe

Run inside the normal GPU test environment from the checkout being measured:

```bash
uv run --no-sync python tests/performance_tests/shell_test_utils/determinism/benchmark.py \
  --output /tmp/dense-paired --recipe dense --gpus 8
```

Presets are `dense`, `moe`, and `hybrid`, using BF16 mock-data training. The
hybrid preset includes Mamba, attention, and MLP layers. Presets require an even
GPU count because TP is 2; MoE requires a multiple of four for TP=2 and EP=2.
Additional workloads can supply a launcher command
after `--`; it must honor `DETERMINISM_PERF_MODE`, `DETERMINISM_PERF_LOG_DIR`, and
`DETERMINISM_PERF_TRAIN_ITERS`, and emit the same iteration log contract.

To measure the PR against a base source checkout on the **same allocation**:

```bash
uv run --no-sync python tests/performance_tests/shell_test_utils/determinism/benchmark.py \
  --output /tmp/dense-base-head --recipe dense --gpus 8 \
  --base-checkout /path/to/base-checkout
```

The head's launcher and Python environment run both revisions. This isolates
source changes; dependency/image changes need a separately controlled comparison.
The base checkout is read only. `head_overhead` measures head det/default;
`det_regression` measures head det/base det; `default_regression` measures head
default/base default. This catches slowdowns shared by both execution modes.
`base_overhead` is reported for context and does not gate the proposed change.

Outputs must use a fresh directory. Failed attempts are retained, never
overwritten by a retry. Dirty source trees or unavailable GPU provenance yield
an inconclusive result. Optional profiling through
`DETERMINISM_PERF_PROFILE=1 bash .../perf_breakdown.sh OUT LOGS` runs afterward;
the NVTX range table describes host annotations and must not be summed as GPU
kernel time or used as the latency gate.

The H100 dense recipe remains in L1. Broader MoE/hybrid and GB200 rows are
scheduled at nightly cadence within L1, which the current workflow selects.
Labels that bypass cadence also select those rows. The GPU presets and initial
budgets require runtime validation before treating their reports as baselines.

CPU tests:

```bash
PYTHONPATH=. python -m pytest --confcutdir=tests/unit_tests/determinism_reporting \
  tests/unit_tests/determinism_reporting/test_paired_performance.py
```
