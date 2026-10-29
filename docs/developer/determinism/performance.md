---
orphan: true
---

# Determinism performance measurements

`benchmark.py` reports repeated **unprofiled** training timings for a
deterministic/default pair of runs. It runs only when invoked; the existing
`determinism_perf` CI job (`perf_breakdown.sh`, an Nsight-profiled per-range
breakdown with its own step-time check) is unchanged. JSON and Markdown
artifacts retain all paired runs, per-step samples, revisions, GPU identifiers,
driver/package versions, and effective mode settings. A timing pass does not
establish determinism.

## Protocol

The default is three deterministic/default pairs, each with 20 warmup steps and
50 measured steps. Every arm starts a fresh process; pair order alternates to
reduce order bias. Both arms use identical model dimensions, input seed,
parallelism, and Python dependencies. Determinism-owned environment variables
are explicitly reset before launch, including in the default arm.
`TRITON_CACHE_AUTOTUNING` is unset in both arms: deterministic SSM uses its
pinned-config fallback. An inherited cache directory or explicit
`TRITON_AUTOTUNE_BLOCK_*` overrides are retained equally and recorded. Cached
SSM autotuning is a separate policy that needs its own controlled benchmark.

The report uses the median iteration time of each run, then ratios within each
pair. A paired bootstrap interval describes run-to-run uncertainty. Fewer than
three pairs or an interval that straddles the limit produces `inconclusive`
(exit 2), not a pass. A confirmed regression or invalid/incomplete timing log
exits 1. A valid pass exits 0. Three pairs are a minimum, not a guarantee that
rare noise is characterized; increase the count when calibrating.

Training comparisons default to a 1.35 deterministic/default limit. The optional
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
after `--`. The tokens `{mode}` (`det` or `default`), `{log_dir}` and
`{train_iters}` are replaced for each arm; the command must write the same
iteration log contract to that directory.

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
an inconclusive result. For attribution, `perf_breakdown.sh` produces the
Nsight per-NVTX-range breakdown; the range table describes host annotations and
must not be summed as GPU kernel time or used as a latency gate. The GPU presets
and default limits need runtime validation on the target hardware before their
reports are treated as baselines.

## Kernel leaderboard pilot

The operator pilot covers `bias_swiglu`, `weighted_swiglu`, and
`weighted_squared_relu` from the production fusion modules. It produces six
rows per precision: separate forward and backward measurements for each case.
BF16 and FP32 activations produce twelve rows in total, with 4096 tokens,
hidden size 8192 and FP32 token weights.
This small selection is not a leaderboard of every registered kernel.

```bash
uv run --no-sync python tests/performance_tests/shell_test_utils/determinism/kernel_leaderboard.py \
  --output /tmp/kernel-leaderboard
```

Each arm starts a fresh process with the requested policy before operator imports.
CUDA events time all GPU work in the operator call. Compilation and warmup are
excluded; backward also excludes forward/graph construction and upstream-gradient
allocation. `autograd.grad` avoids accumulating leaf gradients. The interval can
include launch gaps; it is operator latency, not a sum of individual device-kernel
durations. The report records input shapes/strides/dtypes, CUDA/driver/GPU/package
versions, source revisions and the uninitialized-memory-fill setting. The shared
`kernel_case.py` adapter generates native-dtype inputs with seed 1234 for both
timing and author replay; backward uses an all-ones upstream gradient in both.
SHA-256 fingerprints include every input's exact bytes, shape, stride, dtype
and gradient requirement, retaining positional `None` arguments. Input hashing
and context capture happen before timing. UUIDs identify the actual timing GPU.

Kernel comparisons default to **report only** (`reported`, exit 0), with no
performance pass claim. Missing/invalid timings fail; insufficient pairs, dirty
source or missing GPU provenance are inconclusive. A failing row stays visible
while later rows still run. A new run needs a fresh output directory.

After calibrating a specific case, phase, dtype, shape and GPU, an explicit budget
can gate that row. Base/head runs use the same driver and allocation:

```bash
uv run --no-sync python tests/performance_tests/shell_test_utils/determinism/benchmark.py \
  --output /tmp/swiglu-backward --kernel-case weighted_swiglu --phase backward --gpus 1 \
  --base-checkout /path/to/base-checkout \
  --max-overhead-ratio <calibrated-ratio> --max-regression-ratio <calibrated-ratio>
```

`--tokens`, `--hidden-size`, and `--dtype` select another case configuration.
Keep H100 and GB200 measurements separate. Set the launch environment
explicitly for each platform and retain it in the report. Enforcing calibrated
changed-kernel budgets remains separate from publishing measurements.

### Diagnose a timing difference

Use the Nsight breakdown from `perf_breakdown.sh` for attribution. Benchmark
results use only unprofiled CUDA-event or training-step samples.
