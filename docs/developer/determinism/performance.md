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
`TRITON_CACHE_AUTOTUNING` is unset in both arms: deterministic SSM uses its
pinned-config fallback. An inherited cache directory or explicit
`TRITON_AUTOTUNE_BLOCK_*` overrides are retained equally and recorded. Cached
SSM autotuning is a separate policy that needs its own controlled benchmark.

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

The H100 and GB200 `determinism-kernel-perf.yaml` recipes select this pilot at
nightly cadence in L1 (and when labels bypass cadence). They use one GPU per
measurement and upload `leaderboard.json`, `leaderboard.md`, full per-case
reports, raw samples and logs through the existing assets directory. A cadence
selects eligible jobs; a scheduled or explicit workflow trigger must still run.

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
Keep H100 and GB200 baselines separate. The GB200 recipe uses
`CUDA_DEVICE_MAX_CONNECTIONS=32`, matching its replay bucket; H100 uses 1.
Historical baseline bundles are described below. Enforcing calibrated
changed-kernel budgets remains separate from publishing measurements.

### Diagnose a timing difference

Use `--diagnostics` for a focused investigation on the same case, phase, shape,
precision and allocation policy. It runs the existing CUDA-event timing loop
with an external `nvidia-smi` sampler, then profiles three calls in the same
process. Forward and backward are supported; each backward trace excludes
forward/graph construction and upstream-gradient allocation. All kernels from
each profiled call are retained, including unfused or multi-kernel dispatches.

```bash
TRITON_CACHE_DIR=/tmp/diagnostic-triton TORCHINDUCTOR_CACHE_DIR=/tmp/diagnostic-inductor \
uv run --no-sync python tests/performance_tests/shell_test_utils/determinism/benchmark.py \
  --output /tmp/swiglu-diagnostic --kernel-case bias_swiglu --phase backward --gpus 1 \
  --dtype float32 --pairs 3 --warmup 50 --steps 100 --diagnostics
```

Each arm retains `diagnostics/diagnostics.json`, the original event samples,
three Chrome traces, telemetry CSV/stderr, CPU affinity and thread settings,
and hashes of selected compiler-cache files before/after profiling. Cache roots
are read from the two explicit environment variables; the tool does not change
them. Unset, empty or unreadable inventories produce
`compiler_cache_unchanged: null`. Missing telemetry is recorded as unavailable,
and a timing/profile failure preserves partial diagnostics and fails the run.

The report has status **`diagnostic`** (exit 0 means collection succeeded), and
both raw results and report measurements carry `diagnostic_only: true`.
Diagnostic runs cannot use performance limits, supply author performance
evidence, or be published as baselines. Default benchmark runs are unchanged.

External sampling can affect timing. The sparse telemetry interval includes
compilation, warmup, timing, hashing and profiling, so it does not isolate the
measured event samples. Post-timing profiles observe later calls; unchanged
cache hashes do not prove the actual earlier dispatch identity. Shared cache
changes can come from another process. Keep profiles and event intervals
distinct, and do not infer a historical cause or calibrated speed limit from
these diagnostics alone.

## Join author checks and phase timings

Run the coverage producer and performance driver from the **same clean source
revision**, containing both features. Reports from separate PR heads cannot be
joined. The shared adapter records its own source hash, input fingerprints,
strict Torch policy (including warn-only and memory-fill settings), autocast/TF32/
cuDNN settings, CUDA/driver/GPU details, package versions and environment overrides.
Default and deterministic timing arms must differ only in the declared policy;
old reports lacking this contract remain ineligible for author evidence.

After collecting the coverage report and the twelve-row leaderboard:

```bash
uv run --no-sync python tests/performance_tests/shell_test_utils/determinism/author_evidence.py \
  --coverage /tmp/replay/coverage.json \
  --performance /tmp/kernel-leaderboard/leaderboard.json \
  --revision "$(git rev-parse HEAD)" \
  --output /tmp/author-evidence.json
```

Repeat `--performance` to supply separate `benchmark.json` files. Use one attempt
and one hardware/configuration context per bundle. The join retains every
manifest-required case, requires passing reference/sensitivity/replay evidence
on every replay rank, and requires one matching forward and one backward report.
It rejects duplicates instead of selecting a favorable retry. Raw samples,
medians and paired comparisons are checked again, and every arm of a phase must
use the same GPU UUID. Local activation replay may be replicated across multiple
ranks while timing uses one GPU; this adapter contains no collectives and makes
no inference about distributed performance. Other operators need explicit adapters.

`evidence_complete` and performance status answer different questions. Complete
head-only or unbudgeted timings are `not_gated`, even if det/default overhead met
a configured limit. A performance pass also requires a common base revision for
both phases, complete base/head arms, and passing explicit overhead and revision
limits. This catches slowdowns shared by default and deterministic modes. The
limits still need independently reviewed calibration; their presence is not proof
that the budgets were calibrated.

The CLI writes JSON and Markdown plus hashes of its input artifacts before
returning. Exit 0 means complete, nonfailing evidence; exit 1 means a numerical or
performance failure; exit 2 means missing/incompatible/uncertain evidence. Add
`--require-performance-pass` to also return 2 for an unbudgeted bundle. Existing
reports are not overwritten. Combining compatible coverage/performance CI jobs
and calibrated PR-wide enforcement still needs the combined GPU workflow.

## Publish and verify a historical baseline

The existing CI log uploader already includes the kernel leaderboard, separate
benchmark reports, raw timing files and launcher logs. Download the coverage and
performance artifacts from the same clean source revision and compatible runtime
context. Keep one attempt per directory; do not combine retries or allocations.

```bash
python tests/performance_tests/shell_test_utils/determinism/baseline.py publish \
  --coverage /tmp/coverage-logs/determinism-coverage.json \
  --leaderboard /tmp/perf-logs/kernel-leaderboard/leaderboard.json \
  --revision <full-source-revision> \
  --origin <CI-run-or-execution-reference> \
  --store /shared/determinism-baselines
```

Publication recomputes the author/timing join for every declared requirement and
requires a matching forward and backward report. It checks every separate
`benchmark.json` against the leaderboard and every raw `kernel.json` against its
embedded measurements. All rows must share the source, measurement protocol and
timing GPU. Missing files, duplicate attempts, extra rows, incomplete evidence
and failed numerical or performance checks prevent publication. No missing phase
is filled from a nearby configuration or a different run.

The store must be outside the downloaded artifact directory. Each baseline lives
under the SHA-256 of its `baseline.json` manifest. It contains the original
coverage aggregate, full performance reports, raw timing files and logs, with
relative file names, sizes and hashes. Original runner paths inside reports are
retained as provenance; verification does not require those paths to exist.
The manifest also retains the recomputed author evidence. Identical publication
reuses a verified bundle; existing baselines are never refreshed in place.

After copying or archiving a bundle, verify it against the saved identifier:

```bash
python tests/performance_tests/shell_test_utils/determinism/baseline.py verify \
  /downloaded/baseline --expected-id <saved-manifest-sha256>
```

Verification checks the complete file inventory and recomputes the numerical
evidence/timing join again. Hashes check integrity, not execution authenticity;
the origin is supplied by the publisher and should reference the actual CI run.
The coverage aggregate retains observations and checks; this tool does not
reconstruct it from pytest shards or independently rerun GPU comparisons.

Unbudgeted bundles remain `not_gated`. Publication never assigns limits, promotes
historical timings into a performance pass, or makes cross-allocation timings
equivalent to paired base/head measurements. H100 and GB200 remain separate
contexts. Retain the content-addressed store in durable storage; the ordinary CI
log artifact's retention period alone does not provide permanent publication.

CPU contract tests use explicitly synthetic GPU metadata and timings. They do
not establish hardware latency, correctness, replay coverage or usable budgets.

## Consume CI artifacts

The shared GitHub test action stamps only the latest kernel replay bucket and
kernel performance pilot. Their existing log artifacts retain the repository,
actual checked-out revision, run ID, run attempt, platform, test case and raw
producer outcome/exit code. Their names include the run and attempt so a rerun
does not silently consume an earlier attempt. Other log names are unchanged.

When the actual integration matrix selects `determinism_kernel_perf`, the CPU
`cicd-determinism-baselines` job downloads those current-attempt artifacts into
separate named directories. For each selected platform it requires exactly one
successful replay producer and one successful timing producer, checks all source
identities and the observed GPU, then invokes the baseline publisher above.
Failed/missing producers, mismatched sources or duplicate uploads cannot pass.
A lone upload retry is accepted; two available copies remain ambiguous and must
be investigated instead of selecting the more favorable report. Data from a
platform whose performance pilot was not selected receive no verification credit.
The consumer executes the checked-out verifier, never code from log artifacts.

The job retains `report.json`, `report.md` and each successfully verified scoped
bundle in `determinism-baselines-<run>-a<attempt>`, including diagnostic reports on
failure. The final CI gate waits for artifact verification. A complete artifact
check does not turn `not_gated` timing into a performance pass. The consumer needs
the measured coverage producer from the companion coverage change (#7317);
registration-only reports cannot substitute for it.

To reproduce the consumer after downloading the original named artifacts:

```bash
python tests/performance_tests/shell_test_utils/determinism/ci_artifacts.py collect \
  --artifacts /tmp/determinism-inputs --output /tmp/determinism-candidates \
  --repository NVIDIA/Megatron-LM --revision <full-source-revision> \
  --run-id <run-id> --attempt <run-attempt> --platform dgx_h100 --platform dgx_gb200
```

Pass only platforms selected by that run's actual performance matrix. The output
directory must be empty and outside the downloaded inputs. A partial rerun that
does not rerun both required producers cannot reuse older evidence and remains
unverified. Run both producers in the new attempt to obtain a complete candidate.
GitHub artifact retention is temporary; durable storage, reviewed promotion and
calibrated performance budgets remain separate responsibilities. CPU transport
tests do not establish successful protected CI execution or GPU acceptance.

CPU tests:

```bash
PYTHONPATH=. python -m pytest --confcutdir=tests/unit_tests/determinism_reporting \
  tests/unit_tests/determinism_reporting/test_paired_performance.py
```
