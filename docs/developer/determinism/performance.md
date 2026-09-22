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

## Join author checks and phase timings

Run the coverage producer and performance driver from the **same clean source
revision**, containing both features. Reports from separate PR heads cannot be
joined. The shared adapter records its own source hash, input fingerprints,
strict Torch policy (including warn-only and memory-fill settings), autocast/TF32/
cuDNN settings, CUDA/driver/GPU details, package versions and environment overrides.
Default and deterministic timing arms must differ only in the declared policy.

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
reports are not overwritten.

## Publish and verify a historical baseline

A kernel leaderboard directory holds `leaderboard.json`, the separate per-case
benchmark reports, raw timing files and launcher logs. Use it together with the
coverage report from the same clean source revision and compatible runtime
context. Keep one attempt per directory; do not combine retries or allocations.

```bash
python tests/performance_tests/shell_test_utils/determinism/baseline.py publish \
  --coverage /tmp/coverage-logs/determinism-coverage.json \
  --leaderboard /tmp/perf-logs/kernel-leaderboard/leaderboard.json \
  --revision <full-source-revision> \
  --origin <run-reference> \
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
the origin is supplied by the publisher and should identify the run that produced
the inputs.
The coverage aggregate retains observations and checks; this tool does not
reconstruct it from pytest shards or independently rerun GPU comparisons.

Unbudgeted bundles remain `not_gated`. Publication never assigns limits, promotes
historical timings into a performance pass, or makes cross-allocation timings
equivalent to paired base/head measurements. H100 and GB200 remain separate
contexts. Keep the content-addressed store in durable storage; temporary run
outputs alone do not provide permanent publication.

### Compare retained calibration runs

Use pinned, verified, unbudgeted bundles to review variation across runs:

```bash
python tests/performance_tests/shell_test_utils/determinism/calibration.py \
  --baseline /store/first-run <first-manifest-sha256> \
  --baseline /store/second-run <second-manifest-sha256> \
  --output /reports/calibration.json
```

The command rechecks every bundle's files and author/timing join. It rejects
duplicate identifiers, identical measurements republished with different metadata,
and budgeted results whose selection could hide failed measurements. Invalid input
fails the report instead of silently selecting the remaining passing records.

Comparison groups retain exact source revisions, cases, shapes, inputs, precision,
adapter, runtime settings and measurement protocol. H100/GB200, changed drivers,
software versions, base revisions and phase protocols remain separate. Host names,
GPU UUIDs and checkout locations remain in each run's provenance; they do not
split a group. Counts of distinct timing GPUs do not establish statistical
independence or prove separate scheduler allocations.

Cache paths match by default. For runs using different explicit cache directories,
`--compare-cache-locations` groups those locations while preserving the actual
paths in every run. Unset and explicit cache locations still remain separate.
This option does not establish equal cache contents, generated code or dispatch.
Other runtime settings continue to require exact equality.

Each bundle contributes one median paired ratio. JSON retains each run's paired
ratios and within-run intervals; Markdown shows the minimum, median, maximum and
observed range across runs. Default/deterministic overhead stays separate from
paired head/base regression. Raw event samples are never pooled to manufacture
more independent observations, and the observed range is not a confidence or
prediction bound. One-run groups are explicitly marked in JSON.

The result remains `report_only` / `not_gated`. The tool does not choose budgets,
approve promotion or replace production-recipe measurements. Inputs are selected
published bundles, not a complete survey of executions. Preserve unsuccessful
attempts alongside them when reviewing calibration and unexplained variation.

CPU contract tests use explicitly synthetic GPU metadata and timings. They do
not establish hardware latency, correctness, replay coverage or usable budgets.

## Time actual captured collectives

`benchmark_collectives.py` measures the six direct TP/SP mappings supported by
the [recipe capture adapter](recipe-coverage.md), using its actual rank-local
input and upstream-gradient bytes. This optional adapter needs the capture and
replay producer (#7260/#7317) and the early startup API (#7419). It runs on the
capture's original single-node allocation; a capture from a different physical
GPU assignment cannot supply these measurements.

After capture and replay have completed at the clean head revision, run the
parent outside `torchrun`, from that head checkout:

```bash
python tests/performance_tests/shell_test_utils/determinism/benchmark_collectives.py \
  --capture /results/recipe/capture \
  --evidence /results/recipe/coverage.json \
  --base-checkout /checkouts/base \
  --output /results/recipe/collective-timing \
  --pairs 3 --warmup 20 --steps 50
```

The capture must have matching passing replay, reference and sensitivity
observations on every rank for each selected event. `--event-indices 0 1`
selects explicit entries from the manifests; omission measures every event.
A `forward` event measures forward, and its `forward_backward` event measures
backward with the captured upstream gradient. The latter's forward graph setup
is outside timing. Both phases restore the input and gradient before every
sample, including their original strides and storage offsets.

Each source/policy arm starts fresh processes and process groups; pair order
alternates. Explicit group membership, stream priority, NCCL options, input
hashes, UUIDs and non-policy settings stay fixed. The deterministic arm retains
the captured policy. The default arm disables Torch/cuDNN deterministic
selection, removes NCCL_ALGO/CUBLAS_WORKSPACE_CONFIG and enables the existing
default TE/Mamba/causal-convolution settings. Memory fill, TF32, Triton caching
and other NCCL overrides stay fixed and are recorded. Autocast and cuDNN
benchmark captures currently require a different adapter. The head's timing
helpers run both sources; production mapping imports are checked against the
selected checkout. Matching correctness evidence applies to deterministic head
execution, not to default or baseline accuracy.

Each communicator is initialized with a group barrier before operator warmup.
The report retains requested options and the actual settings before and after
initialization. PyTorch resolves NCCL's undefined `blocking` field to `0` or `1`
according to the explicit configuration/environment; only that declared
resolution is accepted. Stream priority, CTA limits and every other option stay
exact. Measured signatures record the resolved settings alongside the original
capture signature. See [ProcessGroupNCCL initialization](https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp).

Rank alignment barriers, input restoration, graph setup and warmup are excluded
from CUDA-event intervals. These direct synchronous c10d mappings join NCCL
completion to the calling stream, so its end event includes completion; see
[PyTorch's collective stream semantics](https://docs.pytorch.org/docs/2.14/distributed.html#synchronous-and-asynchronous-collective-operations).
Intervals include host launch gaps and arrival skew. They measure isolated
operator phases, including identity phases, not pure NCCL kernel durations,
overlap or whole-model throughput.

Every arm retains raw per-rank JSON and a log. For each event and captured
group, the report takes the maximum rank latency at each aligned sample index,
then its median. Paired bootstrap intervals use these independent process-pair
ratios; ranks and samples are never pooled as independent repetitions. JSON and
Markdown retain each group's result, head overhead and optional base/head
regressions. Incomplete ranks, changed source/runtime, modified captures or
missing accuracy evidence fail and leave the partial attempt for diagnosis.

Ratios are report-only unless explicit reviewed `--max-overhead-ratio` and/or
`--max-regression-ratio` limits are supplied; fewer than three pairs remain
inconclusive. Local-activation leaderboard and calibration reports retain their
own format. Use the explicit collective publication and artifact-consumer paths
below for this report kind. Reviewed baseline promotion and production-recipe
performance acceptance remain separate work.
